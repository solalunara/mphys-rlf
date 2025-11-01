import sys
import os
import bdsf;
from astropy.io import fits;
from astropy.io.fits import ImageHDU;
import bdsf.image
import numpy as np;
import matplotlib.pyplot as plt;
import math;
from pathlib import Path, PurePath;
import multiprocessing;
import multiprocessing.pool;

# Add the src directory to Python path so we can import modules
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_dir)

import utils.logging;

logger = utils.logging.get_logger(__name__);

#Neccesary pool extention - PyBDSF uses daemon processes but only sometimes, and we want to batch the files themselves
#Courtesy of https://stackoverflow.com/questions/52948447/error-group-argument-must-be-none-for-now-in-multiprocessing-pool
class NonDaemonPool(multiprocessing.pool.Pool):
    def Process(self, *args, **kwds):
        proc = super(NonDaemonPool, self).Process(*args, **kwds)

        class NonDaemonProcess(proc.__class__):
            """Monkey-patch process to ensure it is never daemonized"""
            @property
            def daemon(self):
                return False

            @daemon.setter
            def daemon(self, val):
                pass

        proc.__class__ = NonDaemonProcess
        return proc




class ImageAnalyzer:
    """
    A class to analyze images of radio galaxies using PyBDSF, with LOFAR defaults

    Parameters
    ----------
    subdir : str
        The subdirectory appended to all root directories, to separate different use cases,
        i.e. files are read from "[fits_input_dir]/[subdir]/\*\*.fits",
        catalogs are written to "[catalog_dir]/[subdir]/\*\*.fits",
        and images are written to "[img_dir]/[subdir]/\*\*.fits",
        where "\*\*" implies the files are recursively searched for, allowing the data to be segmented into folder bins

    catalog_dir : str | Path = "pybdsf_catalogs/"
        Root directory of all catalogs. Catalogs are written to "[catalog_dir]/[subdir]/\*\*.fits"

    img_dir : str | Path = "fits_images/reconstructed"
        Root directory of all images exported by PyBDSF. Images are written to "[img_dir]/[subdir]/[img_type]/\*\*.fits"

    fits_input_dir : str | Path = "fits_images/"
        Root directory of all fits input files. Images are taken from "[fits_input_dir]/[subdir]/\*\*.fits",
        where the postfix after subdir is used as the postfix for img_dir and catalog_dir
    
    process_args : dict = LOFAR_process_arg_defaults
        Arguments to pass to bdsf.process_image. Refer to PyBDSF documentation for an exhaustive list.
        If none, default values for the project are filled from LOFAR:
            beam = (0.00166667, 0.00166667, 0.0),
            thresh_isl = 5,
            thresh_pix = 0.5,
            mean_map = "const",
            rms_map = True,
            thresh = "hard",
            frequency = 144e6

    catalog_args : dict | None = dict( format='fits', catalog_type='srl', clobber=True )
        Arguments to pass to the bdsf.write_catalog function, or None to not write a catalog.
        Refer to PyBDSF documentation for an exhaustive list.

    export_img_args: list[dict] = [ dict( img_type='gaus_model', clobber=True ) ]
        Arguments to pass to the bdsf.export_image function. The array entries correspond to calls to the function,
        allowing multiple different calls, or an empty array to not export any images. The default will export
        one image - the reconstruction
    """

    LOFAR_process_arg_defaults = dict(
        beam = (0.00166667, 0.00166667, 0.0),
        thresh_isl = 5,
        thresh_pix = 0.5,
        mean_map = "const",
        rms_map = True,
        thresh = "hard",
        frequency = 144e6
    );

    def __init__( self, 
                  subdir: str | PurePath, 
                  fits_input_dir: str | Path = "fits_images/",
                  catalog_dir: str | Path = "pybdsf_catalogs/",
                  img_dir: str | Path = "fits_images/exported/",
                  process_args: dict = LOFAR_process_arg_defaults,
                  catalog_args: dict | None = dict(
                    format='fits',
                    catalog_type='srl',
                    clobber=True
                  ),
                  export_img_args: list[dict] = [
                      dict( img_type='gaus_model', clobber=True )
                  ] ):

        #arrays and dicts need to be copied from templates for memory safety
        self.process_args = process_args.copy();
        self.catalog_args = catalog_args.copy();
        self.export_img_args = [ d.copy() for d in export_img_args ];

        self.catalog_dir = catalog_dir if isinstance( catalog_dir, Path ) else Path( catalog_dir );
        self.img_dir = img_dir if isinstance( img_dir, Path ) else Path( img_dir );
        self.fits_input_dir = fits_input_dir if isinstance( fits_input_dir, Path ) else Path( fits_input_dir );
        self.subdir = subdir if isinstance( subdir, PurePath ) else PurePath( subdir );
    
    def GetUnwrappedFITSInputList( self, path: Path | None = None ):
        """
        Recurse through all files in path and unwrap all fits files into a single list,
        useful for multiprocessing

        Parameters
        ----------
        path: Path | None
            The path to unwrap. None defaults to "[fits_input_dir]/[subdir]"

        Returns
        -------
        list[ Path ]
            An unwrapped list of all fits files in path, or the path itself if it is a fits file
        """
        if path is None:
            path = self.fits_input_dir / self.subdir;
        if path.is_dir():
            unwrapped_sublist = [];
            for iter_file in path.iterdir():
                result = self.GetUnwrappedFITSInputList( iter_file );
                if isinstance( result, list ):
                    unwrapped_sublist = unwrapped_sublist + result;
                elif result is not None:
                    unwrapped_sublist.append( result );
            return unwrapped_sublist;
        elif path.suffix == ".fits":
            return path;
        return None;

    
    def AnalyzeAllFITSInInput( self ):
        """
        Recursively analyze all of "[fits_input_dir]/[subdir]/\*\*.fits"

        Spawns in as many processes as the environment variable NUM_CPUS if set, or if not set
        spawns in one process for each top-level bin in [subdir] for parallel processing
        """
        n_cpus = os.environ[ "N_CPUS" ];
        input_subdir = self.fits_input_dir / self.subdir;
        if n_cpus is None:
            subdirectories = [];
            for subdirectory in input_subdir.iterdir():
                subdirectories.append( subdirectory );
            files = subdirectories;
        elif isinstance( n_cpus, str ):
            n_cpus = int( n_cpus );

        if isinstance( n_cpus, int ):
            files = self.GetUnwrappedFITSInputList( input_subdir );

        p = NonDaemonPool( processes=n_cpus );
        p.map( self.AnalyzeFITSInPath, files );
        
        
    
    def AnalyzeFITSInPath( self, path: Path | str ):
        """
        Recursive function to analyze all files under a given path

        Parameters
        ----------
        path : Path | str
            the path to analyze, either to a folder which will be analyzed recursively or to a file
        """
        if not isinstance( path, Path ):
            path = Path( path );

        if path.is_dir():
            logger.debug( "Entering directory %s", str( path ) );
            for sub_path in path.iterdir():
                self.AnalyzeFITSInPath( sub_path );
        else:
            if path.suffix == ".fits":
                #First see if we have any work to do
                postfix = path.parts[ (path.parts.index( str( self.subdir ) ) + 1 ): ];
                write_catalog = self.catalog_args is not None;
                if write_catalog:
                    catalog_outfile = self.catalog_dir / self.subdir.joinpath( *postfix );
                    if catalog_outfile.exists():
                        write_catalog = False;
                local_export_img_args = [];
                for single_image_args in self.export_img_args:
                    image_outfile = self.img_dir / self.subdir / PurePath( single_image_args[ "img_type" ] ).joinpath( *postfix );
                    if not image_outfile.exists():
                        local_export_img_args.append( single_image_args.copy() );
                
                if not write_catalog and len( local_export_img_args ) == 0:
                    print( f"Skipping {path}, no work to do" );
                    return; #nothing to do
                print( f"Processing {path}:" );

                #Something to do, process the image
                #note - single star for array expansion, double star for dict expansion
                img: bdsf.image.Image = bdsf.process_image(
                    str( path ),
                    **self.process_args
                );
                for single_image_args in local_export_img_args:
                    image_outfile = self.img_dir / self.subdir / PurePath( single_image_args[ "img_type" ] ).joinpath( *postfix );
                    image_outfile.parent.mkdir( parents=True, exist_ok=True );
                    img.export_image( outfile=str( image_outfile ), **single_image_args );
                if write_catalog and self.catalog_args is not None:
                    catalog_outfile = self.catalog_dir / self.subdir.joinpath( *postfix );
                    catalog_outfile.parent.mkdir( parents=True, exist_ok=True );
                    img.write_catalog( outfile=str( catalog_outfile ), **self.catalog_args );
    
    def SaveImageToFITS( self, image: np.ndarray, postfix: str ):
        """
        Save a numpy 2d array to a fits file under "[fits_input_dir]/[subdir]/"

        Parameters
        ----------
        image : np.ndarray (2D)
            the pixel values that represent the image (should be 80x80)

        postfix : str
            postfix for the fits file. Can either be the name of the fits file (e.g. "example.fits") or the name
            and location under "[fits_input_dir]/[subdir]/" to store it in (e.g. "example_bin/example.fits")
        """
        hdu = fits.PrimaryHDU( image );
        hdu.header[ "CTYPE1" ] = "RA---TAN";
        hdu.header[ "CTYPE2" ] = "DEC--TAN";
        hdu.header[ "CDELT1" ] = 1.5 * 0.00027778;
        hdu.header[ "CDELT2" ] = 1.5 * 0.00027778;
        hdu.header[ "CUNIT1" ] = "deg";
        hdu.header[ "CUNIT2" ] = "deg";
        hdul = fits.HDUList( [ hdu ] );
        outfile = self.fits_input_dir / self.subdir.joinpath( postfix );
        outfile.parent.mkdir( parents=True, exist_ok=True );
        hdul.writeto( str( outfile ), overwrite=True );

    def AnalyzeImage( self, image: np.ndarray, postfix: str ):
        """
        Save a numpy 2d array to a fits file under "[fits_input_dir]/[subdir]/" and analyze it, storing
        the output in "[catalog_dir]/[subdir]/[postfix]" or "[img_dir]/[subdir]/[img_type]/[postfix]" depending
        on ImageAnalyzer parameters

        Parameters
        ----------
        image : np.ndarray (2D)
            the pixel values that represent the image (should be 80x80)

        postfix : str
            postfix for the fits file. Can either be the name of the fits file (e.g. "example.fits") or the name
            and location under "[fits_input_dir]/[subdir]/" to store it in (e.g. "example_bin/example.fits")
        """
        self.SaveImageToFITS( image, postfix );

if __name__ == "__main__":
    dataset_analyzer = ImageAnalyzer( "dataset" );
    generated_analyzer = ImageAnalyzer( "generated" );

    #dataset_analyzer.AnalyzeAllFITSInInput();
    generated_analyzer.AnalyzeAllFITSInInput();