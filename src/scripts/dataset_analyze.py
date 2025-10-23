import sys
import os
import bdsf;
from astropy.io import fits;
from astropy.io.fits import ImageHDU;
import numpy as np;
import matplotlib.pyplot as plt;

# Add the src directory to Python path so we can import modules
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_dir)


img = bdsf.process_image( 
    "fits_images/dataset/0-10000/image" + str(0) + ".fits", 
    beam = (0.00166667, 0.00166667, 0.0),
    thresh_isl = 5,
    thresh_pix = 0.5,
    mean_map = "const",
    rms_map = True,
    thresh = "hard",
    frequency = 144e6 );
img.show_fit();
img.export_image( img_type='gaus_model', outfile="fits_images/reconstructed/dataset/model" + str(0) + ".fits", clobber=True );


with fits.open( "fits_images/reconstructed/dataset/model" + str(0) + ".fits" ) as hdul_model:
    model_image_data = hdul_model[ 0 ].data;
with fits.open( "fits_images/dataset/0-10000/image" + str(0) + ".fits" ) as hdul_model:
    original = hdul_model[ 0 ].data;


fig = plt.figure( figsize=(8, 8) );
gs = fig.add_gridspec(1, 3,
                    left=0.05, right=0.95, bottom=0.1, top=0.95,
                    wspace=0.5, hspace=0.5);
ax1 = fig.add_subplot( gs[ 0, 0 ] );
ax2 = fig.add_subplot( gs[ 0, 1 ] );
ax3 = fig.add_subplot( gs[ 0, 2 ] );
ax1.set_title( 'Original' );
ax2.set_title( 'Reconstruction' );
ax3.set_title( 'Difference' );
img1 = ax1.imshow( original[ :, : ] )
img2 = ax2.imshow( model_image_data[ 0, 0, :, : ] );
img3 = ax3.imshow( model_image_data[ 0, 0, :, : ] - original[ :, : ] );
img1.set_clim( 0, 1 );
img2.set_clim( 0, 1 );
img3.set_clim( 0, 1 );
plt.show();
