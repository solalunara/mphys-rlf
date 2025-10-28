import sys
import os
import bdsf;
from astropy.io import fits;
from astropy.io.fits import ImageHDU;
import numpy as np;
import matplotlib.pyplot as plt;
import math;
from pathlib import Path, PurePath;

# Add the src directory to Python path so we can import modules
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_dir)

fits_images = Path( "fits_images/" );
fits_images_dataset = fits_images / "dataset"

for fits_images_bin in fits_images_dataset.iterdir():
    for fits_image in fits_images_bin.iterdir():
        fits_image_rawstring = str( fits_image );
        img = bdsf.process_image( 
            fits_image_rawstring,
            beam = (0.00166667, 0.00166667, 0.0),
            thresh_isl = 5,
            thresh_pix = 0.5,
            mean_map = "const",
            rms_map = True,
            thresh = "hard",
            frequency = 144e6 );
        outfile = "pybdsf_catalogs/" + fits_image_rawstring[ len( str( fits_images ) ): ];
        os.makedirs( os.path.dirname( outfile ), exist_ok=True );
        img.write_catalog( outfile=outfile, format='fits', catalog_type='srl', clobber=True );


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
