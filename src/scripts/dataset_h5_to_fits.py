import sys
import os
import bdsf;
from astropy.io import fits;
from astropy.io.fits import ImageHDU;
import numpy as np;
import matplotlib.pyplot as plt;
import h5py;
import math;
from pathlib import Path, PurePath;
import shutil;

# Add the src directory to Python path so we can import modules
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_dir)

class H5ToFitsConverter:
    def __init__( self, h5_file: str | Path, fits_output_dir: str | Path ):
        if h5_file is not Path:
            h5_file = Path( h5_file );
        if fits_output_dir is not Path:
            fits_output_dir = Path( fits_output_dir );
        
        self.h5_file = h5_file;
        self.fits_output_dir = fits_output_dir;

    def ConvertLOFAR( self, *bin_sizes: list[int] ):
        """
        A highly non-general method to convert the LOFAR h5 dataset used in this project into fits files.
        Outputs fits files to self.fits_output_dir, and groups the images into directories by descending sizes.

        Parameters
        ----------
        *bin_sizes : list[ int ]
            Directory bins to sort the images into for ease of use. Defaults to [] which is no bins.

            For example: bin_sizes array of [10000, 1000, 3000] would be sorted to [10000, 3000, 1000] and files would be stored as

            00010 -> fits_output_dir/0-9999/0-2999/0-999/image10.fits

            09150 -> fits_output_dir/0-9999/9000-9999/9100-9199/image9150.fits

            12120 -> fits_output_dir/10000-19999/12000-14999/12100-12199/image12120.fits

            As can be seen with 9150, bins are funneled such that the upper bound of an inner bin is always less than or equal to the upper bound of all
            its outer bins, and the opposite with the lower bound of an inner bin.
        """
        bin_sizes = sorted( bin_sizes, reverse=True );
        with h5py.File( str( self.h5_file ), 'r' ) as h5:
            images = h5[ 'images' ];

            for i in range( images.shape[ 0 ] ):
                image = images[ i ];

                # the images in the dataset *are* selected by the process in the paper but *are not* scaled 0-1
                # here we do that scaling
                im_max = np.max( image );
                im_min = np.min( image );
                if im_min < 0:
                    raise ValueError( "Images not preprocessed to remove negative values" );
                image = ( image - im_min ) / ( im_max - im_min );


                hdu = fits.PrimaryHDU( image );
                hdu.header[ "CTYPE1" ] = "RA---TAN";
                hdu.header[ "CTYPE2" ] = "DEC--TAN";
                hdu.header[ "CDELT1" ] = 1.5 * 0.00027778;
                hdu.header[ "CDELT2" ] = 1.5 * 0.00027778;
                hdu.header[ "CUNIT1" ] = "deg";
                hdu.header[ "CUNIT2" ] = "deg";
                hdul = fits.HDUList( [ hdu ] );

                #Create bins based on bin_sizes
                filename = self.fits_output_dir;
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
                filename.parent.mkdir( parents=True, exist_ok=True ); #ensure path exists
                hdul.writeto( filename, overwrite=True );

if __name__ == "__main__":

    fits_dataset_path = Path( "fits_images/dataset" );

    #Recreate the dataset if it exists to make sure it has the right bin configuration
    if fits_dataset_path.exists():
        shutil.rmtree( fits_dataset_path );

    converter = H5ToFitsConverter( "image_data/LOFAR/LOFAR_Dataset.h5", "fits_images/dataset" );
    converter.ConvertLOFAR( 10000, 1000, 100 );
