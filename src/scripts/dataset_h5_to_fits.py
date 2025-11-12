# This file was created by Ashley and Luna and defines functions neccesary to convert the LOFAR h5 dataset
# that comes with the project into FITS files needed for PyBDSF, as well as a more specific function which implements
# the first function with the files and parameters presented in utils.paths and utils.parameters

# This file also provides a main function which forces a flush of the fits images directory and remakes it according
# to the command line arguments, or utils.parameters if none are passed

import sys;
from astropy.io import fits;
from astropy.io.fits import ImageHDU;
import numpy as np;
import matplotlib.pyplot as plt;
import h5py;
import math;
from pathlib import Path, PurePath;
import shutil;
import utils.parameters;
import utils.paths;
from tqdm import tqdm;
import files.dataset;
from files.dataset import LOFAR_DATA_PATH;
from utils.distributed import DistributedUtils;

def convert_LOFAR_h5_to_fits( lofar_data_h5: Path, fits_output_dir: Path, cutoff: int | None = utils.parameters.LOFAR_FITS_COUNT_CUTOFF, bin_sizes: list[ int ] | None = None ):
    """
    A method to convert the LOFAR h5 dataset used in this project into fits files. Outputs fits files to 
    fits_output_dir, and groups the images into directories by descending sizes.

    Parameters
    ----------
    lofar_data_h5: Path
        The directory to read the LOFAR data from
    fits_output_dir: Path
        The directory to output the FITS files to
    cutoff : int | None = utils.parameters.LOFAR_FITS_COUNT_CUTOFF
        The number of images to convert and export. Default behaviour in utils.parameters.LOFAR_FITS_COUNT_CUTOFF
    bin_sizes : list[ int ] | None = None
        Directory bins to sort the images into for ease of use. Defaults to the value in utils.parameters

        For example: bin_sizes array of [10000, 1000, 3000] would be sorted to [10000, 3000, 1000] and files would be stored as

        00010 -> fits_output_dir/0-9999/0-2999/0-999/image10.fits

        09150 -> fits_output_dir/0-9999/9000-9999/9100-9199/image9150.fits

        12120 -> fits_output_dir/10000-19999/12000-14999/12100-12199/image12120.fits

        As can be seen with 9150, bins are funneled such that the upper bound of an inner bin is always less than or equal to the upper bound of all
        its outer bins, and the opposite with the lower bound of an inner bin.
    """
    files.dataset.download_dataset();

    if bin_sizes is None:
        bin_sizes = utils.parameters.BINS_ARRAY;

    bin_sizes = sorted( bin_sizes, reverse=True );
    with h5py.File( str( lofar_data_h5 ), 'r' ) as h5:
        images = h5[ 'images' ];
    
        images_len = images.shape[ 0 ];
        num_to_convert = min( cutoff, images_len ) if cutoff is not None else images_len;

        for i in tqdm( range( num_to_convert ) ):
            image = images[ i ];

            # the images in the dataset *are* selected by the process in the paper but *are not* scaled 0-1
            # here we do that scaling
            im_max = np.max( image );
            im_min = np.min( image );
            if im_min < 0:
                raise ValueError( "Images not preprocessed to remove negative values" );
            image = ( image - im_min ) / ( im_max - im_min );


            hdu = fits.PrimaryHDU( image );
            hdu.header[ "CTYPE1" ] = "RA---SIN";
            hdu.header[ "CTYPE2" ] = "DEC--SIN";
            hdu.header[ "CDELT1" ] = 1.5 * 0.00027778;
            hdu.header[ "CDELT2" ] = 1.5 * 0.00027778;
            hdu.header[ "CUNIT1" ] = "deg";
            hdu.header[ "CUNIT2" ] = "deg";
            hdu.header[ "FXSCLD" ] = (im_max**(-0.23) - 1)/(-0.23);
            hdul = fits.HDUList( [ hdu ] );

            #Create bins based on bin_sizes
            filename = fits_output_dir;
            #make sure that there are no overextending bounds (e.g. 0-9999/9000-11999)
            lowest_upper_bound =  1e30;
            highest_lower_bound =-1e30;
            for bin_size in bin_sizes:
                lower_bound = int( math.floor( i / bin_size ) * bin_size );
                upper_bound = int( math.ceil( ( i + 1 ) / bin_size ) * bin_size ) - 1;

                if lower_bound < highest_lower_bound:
                    lower_bound = highest_lower_bound; #cut this lower bound to the highest before it
                else:
                    highest_lower_bound = lower_bound; #new highest lower bound

                if upper_bound > lowest_upper_bound:
                    upper_bound = lowest_upper_bound;  #cut this upper bound to the lowest before it
                else:
                    lowest_upper_bound = upper_bound;  #new lowest upper bound

                filename = filename / PurePath( f"{lower_bound}-{upper_bound}" );

            filename = filename / PurePath( f"image{i}.fits" );
            if not filename.exists():
                filename.parent.mkdir( parents=True, exist_ok=True ); #ensure path exists
                hdul.writeto( filename, overwrite=True );

def single_node_convert_LOFAR_h5_to_fits( lofar_data_h5: Path, fits_output_dir: Path, cutoff: int | None = utils.parameters.LOFAR_FITS_COUNT_CUTOFF, bin_sizes: list[int] | None = None ):
    du = DistributedUtils();
    du.single_task_only_forcewait( 'convert_LOFAR_h5_to_fits', convert_LOFAR_h5_to_fits, 0, lofar_data_h5, fits_output_dir, cutoff, bin_sizes );

def validate_LOFAR_fits_images( clean_directory: bool, cutoff: int | None = utils.parameters.LOFAR_FITS_COUNT_CUTOFF, bin_sizes: list[ int ] | None = None ):
    """
    Ensure FITS images from LOFAR exist in accordance with paths laid out in utils.paths

    Parameters
    ----------
    clean_directory : bool
        Whether or not to clean out the fits images directory to ensure bin compliance
    cutoff : int | None = utils.parameters.LOFAR_FITS_COUNT_CUTOFF
        The optional value to cut off conversion at, in terms of number of images converted

    bin_sizes : list[ int ] = None
        Directory bins to sort the images into for ease of use. Defaults to utils.parameters.BINS_ARRAY
        Is only used in the case where the dataset folder does not already exist. Compliance with bin structure is not checked for.

        For example: bin_sizes array of [10000, 1000, 3000] would be sorted to [10000, 3000, 1000] and files would be stored as

        00010 -> fits_output_dir/0-9999/0-2999/0-999/image10.fits

        09150 -> fits_output_dir/0-9999/9000-9999/9100-9199/image9150.fits

        12120 -> fits_output_dir/10000-19999/12000-14999/12100-12199/image12120.fits

        As can be seen with 9150, bins are funneled such that the upper bound of an inner bin is always less than or equal to the upper bound of all
        its outer bins, and the opposite with the lower bound of an inner bin.
    """
    fits_dataset_folder = utils.paths.FITS_PARENT / utils.paths.DATASET_SUBDIR;
    if fits_dataset_folder.exists():
        if clean_directory:
            shutil.rmtree( fits_dataset_folder );
        else:
            return;

    convert_LOFAR_h5_to_fits( LOFAR_DATA_PATH, fits_dataset_folder, cutoff, bin_sizes );

def single_node_validate_LOFAR_fits_images( clean_directory: bool, cutoff: int | None = utils.parameters.LOFAR_FITS_COUNT_CUTOFF, bin_sizes: list[ int ] | None = None ):
    du = DistributedUtils();
    du.single_task_only_forcewait( 'validate_LOFAR_fits_images', validate_LOFAR_fits_images, 0, clean_directory, cutoff, bin_sizes );



if __name__ == "__main__":
    # If this file is run directly, do a cleanup of the fits dataset dir and remake it according to args
    bin_sizes = [];
    for arg in sys.argv[ 1: ]:
        bin_sizes.append( int( arg ) );
    if len( bin_sizes ) == 0:
        bin_sizes = utils.parameters.BINS_ARRAY;
    single_node_validate_LOFAR_fits_images( True, bin_sizes=bin_sizes );