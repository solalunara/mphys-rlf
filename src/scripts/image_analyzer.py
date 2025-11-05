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
from recursive_file_analyzer import RecursiveFileAnalyzer;


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




class ImageAnalyzer( RecursiveFileAnalyzer ):
    """
    A class to analyze images of radio galaxies using PyBDSF, with LOFAR defaults

    Parameters
    ----------
    subdir : str
        The subdirectory appended to all root directories, to separate different use cases,
        i.e. files are read from "[fits_input_dir]/[subdir]/\*\*.fits",
        catalogs are written to "[catalog_dir]/[subdir]/\*\*.fits",
        and images are written to "[img_dir]/[subdir]/[img_type]/\*\*.fits",
        where "\*\*" implies the files are recursively searched for, allowing the data to be segmented into folder bins

    fits_input_dir : str | Path = "fits_images/"
        Root directory of all fits input files. Images are taken from "[fits_input_dir]/[subdir]/\*\*.fits",
        where the postfix after subdir is used as the postfix for img_dir and catalog_dir

    catalog_dir : str | Path = "pybdsf_catalogs/"
        Root directory of all catalogs. Catalogs are written to "[catalog_dir]/[subdir]/\*\*.fits"

    img_dir : str | Path = "fits_images/reconstructed"
        Root directory of all images exported by PyBDSF. Images are written to "[img_dir]/[subdir]/[img_type]/\*\*.fits"

    write_catalog : bool = True
        Whether or not to write a catalog. If true, arguments can be passed to bdsf.write_catalog
        by prefixing them with 'catalog_' in kwargs. For example, the default args are "catalog_type='srl', 
        catalog_clobber=True" which is the equivalent of passing "catalog_type='srl', clobber=True" to 
        bdsf.write_catalog. Of note, the pybdsf argument 'catalog_type' is shortened here to simply 'type' 
        for the avoidance of writing the argument 'catalog_catalog_type'. If 'catalog_catalog_type' is passed 
        as an argument it will be ignored. If false, catalog_ args are ignored. Outfile cannot be specified - see
        catalog_dir or subdir for output directory structure.

    export_images : list[ str ]
        Types of pybdsf images to export. See pybdsf documentation for the types of images that can be exported.
        Arguments can be passed to bdsf.export_image by prefixing them with the image type in export_images followed
        by an underscore. For example, to write pybdsf's 'gaus_model' image, pass 'gaus_model' to export_images as
        a free string parameter, then to overwrite if the file exists with pybdsf's 'clobber' parameter, pass
        "gaus_model_clobber=True" in to kwargs. Outfile cannot be specified - see subdir for output directory structure

    **kwargs : dict
        All arguments to pass to bdsf.export_catalog, bdsf.export_image, and bdsf.process_image. The method for
        formatting arguments to export_catalog and export_image are explained above in write_catalog and *export_images.
        To pass arguments to bdsf.process_image, pass the arguments with the prefix 'process_'. For example, to process
        an image with the defaults used in this project, pass "process_beam = (0.00166667, 0.00166667, 0.0),
        process_thresh_isl = 5, process_thresh_pix = 0.5, process_mean_map = 'const', process_rms_map = True,
        process_thresh = 'hard', process_frequency = 144e6". For PyBDSF, beam and frequency must be present.
    """

    LOFAR_process_arg_defaults = dict(
        process_beam = (0.00166667, 0.00166667, 0.0),
        process_thresh_isl = 5,
        process_thresh_pix = 0.5,
        process_mean_map = "const",
        process_rms_map = True,
        process_thresh = "hard",
        process_frequency = 144e6
    );

    def __init__( self, 
                  subdir: str | PurePath, 
                  fits_input_dir: str | Path = "fits_images/",
                  catalog_dir: str | Path = "pybdsf_catalogs/",
                  img_dir: str | Path = "fits_images/exported/",
                  write_catalog: bool = True,
                  export_images: list[ str ] | None = None,
                  **kwargs: dict ):

        #Ensure all types are paths
        self.catalog_dir = catalog_dir if isinstance( catalog_dir, Path ) else Path( catalog_dir );
        self.img_dir = img_dir if isinstance( img_dir, Path ) else Path( img_dir );
        self.fits_input_dir = fits_input_dir if isinstance( fits_input_dir, Path ) else Path( fits_input_dir );
        self.subdir = subdir if isinstance( subdir, PurePath ) else PurePath( subdir );
        self.write_catalog = write_catalog;
        self.export_images = export_images or [];
        self.logger = utils.logging.get_logger( self.__class__.__name__ );

        #Image Analyzer is a recursive analyzer for fits_input_dir/subdir, with additional utilities for catalog_dir and img_dir
        super().__init__( self.fits_input_dir / self.subdir );

        self.process_args = dict();
        self.catalog_args = dict();
        self.export_img_args = dict(); #elements will be (str, img args dict)
        for img_type in export_images:
            self.export_img_args[ img_type ] = dict( img_type=img_type );

        #Loop through kwargs and sort arguments into catalog, export_img, or process
        for key, val in kwargs.items():
            arg_used = False;
            if key.find( 'catalog_' ) > -1:
                if write_catalog:
                    self.catalog_args[ key[ len( 'catalog_' ): ] ] = val;
                else:
                    self.logger.warning( 'WARNING - argument %s passed with catalog prefix but write_catalog is false', key );
                arg_used = True;
            for img_type in export_images:
                if key.find( f'{img_type}_' ) > -1:
                    self.export_img_args[ img_type ][ key[ len( f'{img_type}_' ): ] ] = val;
                    arg_used = True;
            if key.find( 'process_' ) > -1:
                self.process_args[ key[ len( 'process_' ): ] ] = val;
                arg_used = True;
            if not arg_used:
                self.logger.warning( 'WARNING - argument %s passed but not used (are you passing all neccesary strings for export_images?)' )

        #Clobber by default if nothing passed
        if write_catalog:
            self.catalog_args[ 'clobber' ] = self.catalog_args.get( 'clobber', True );
        for img_type in export_images:
            self.export_img_args[ img_type ][ 'clobber' ] = self.export_img_args[ img_type ].get( 'clobber', True );
    
        #Set process arg defaults to project defaults if nothing passed
        for key, val in ImageAnalyzer.LOFAR_process_arg_defaults.items():
            self.process_args[ key[ len( 'process_' ): ] ] = self.process_args.get( key[ len( 'process_' ): ], val );
 
    
    def AnalyzeAllFITSInInput( self ):
        """
        Recursively analyze all of "[fits_input_dir]/[subdir]/\*\*.fits"

        Spawns in as many processes as the environment variable N_CPUS if set, or if not set
        spawns one process
        """
        n_cpus = os.environ[ "N_CPUS" ] or 1;
        input_subdir = self.fits_input_dir / self.subdir;
        if isinstance( n_cpus, str ):
            n_cpus = int( n_cpus );

        files = self.GetUnwrappedList( input_subdir, 'fits' );

        p = NonDaemonPool( processes=n_cpus );
        p.map( self.AnalyzeFITSAtPath, files );
        
        
    
    def AnalyzeFITSAtPath( self, path: Path | str ):
        """
        Function to analyze a single fits file at a given path

        Parameters
        ----------
        path : Path | str
            the path to the file to analyze
        """
        if not isinstance( path, Path ):
            path = Path( path );

        if path.is_dir():
            self.logger.error( 'ERROR - Cannot analyze %s as fits file, is directory', str( path ) );
        else:
            if path.suffix == ".fits":
                #First see if we have any work to do
                postfix = path.parts[ (path.parts.index( str( self.subdir ) ) + 1 ): ];
                write_catalog = self.write_catalog;
                if write_catalog:
                    catalog_outfile = self.catalog_dir / self.subdir.joinpath( *postfix );
                    if catalog_outfile.exists():
                        write_catalog = False;
                export_images = [];
                for img_type in self.export_images:
                    image_outfile = self.img_dir / self.subdir / PurePath( img_type ).joinpath( *postfix );
                    if not image_outfile.exists():
                        export_images.append( img_type );
                
                if not write_catalog and len( export_images ) == 0:
                    self.logger.info( f"Skipping {path}, no work to do" );
                    return; #nothing to do
                self.logger.info( f"Processing {path}:" );

                #Something to do, process the image
                img: bdsf.image.Image = bdsf.process_image(
                    str( path ),
                    **self.process_args
                );
                for img_type in export_images:
                    image_outfile = self.img_dir / self.subdir / PurePath( img_type ).joinpath( *postfix );
                    image_outfile.parent.mkdir( parents=True, exist_ok=True );
                    img.export_image( outfile=str( image_outfile ), **self.export_img_args[ img_type ] );
                if write_catalog:
                    catalog_outfile = self.catalog_dir / self.subdir.joinpath( *postfix );
                    catalog_outfile.parent.mkdir( parents=True, exist_ok=True );
                    img.write_catalog( outfile=str( catalog_outfile ), **self.catalog_args );
            else:
                self.logger.error( 'ERROR - Cannot analyze %s as fits file, is not fits file', str( path ) );
    
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
        self.AnalyzeFITSAtPath( self.fits_input_dir / self.subdir / postfix );

if __name__ == "__main__":
    dataset_analyzer = ImageAnalyzer( "dataset", export_images=[ 'gaus_model', 'gaus_resid' ] );
    generated_analyzer = ImageAnalyzer( "generated", export_images=[ 'gaus_model', 'gaus_resid' ] );

    dataset_analyzer.AnalyzeAllFITSInInput();
    generated_analyzer.AnalyzeAllFITSInInput();