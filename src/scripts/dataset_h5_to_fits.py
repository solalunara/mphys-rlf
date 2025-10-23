import sys
import os
import bdsf;
from astropy.io import fits;
from astropy.io.fits import ImageHDU;
import numpy as np;
import matplotlib.pyplot as plt;
import h5py;
import math;

# Add the src directory to Python path so we can import modules
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_dir)

with h5py.File( "image_data/LOFAR/LOFAR_Dataset.h5", 'r' ) as LOFAR_h5:
    images = LOFAR_h5[ 'images' ];

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

        lower_bound = int( math.floor( i / 10000 ) * 10000 );
        upper_bound = int( math.ceil( ( i + 1 ) / 10000 ) * 10000 );
        filename = "fits_images/dataset/" + str(lower_bound) + "-" + str(upper_bound) + "/image" + str(i) + ".fits";
        os.makedirs( os.path.dirname( filename ), exist_ok=True );
        hdul.writeto( filename, overwrite=True );

