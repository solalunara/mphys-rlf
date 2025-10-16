import sys
import os
import bdsf;
from astropy.io import fits;
from astropy.io.fits import ImageHDU;
import numpy as np;

# Add the src directory to Python path so we can import modules
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_dir)


import model.sampler
import plotting.image_plots

model_sampler = model.sampler.Sampler();

samples = model_sampler.quick_sample(
    "LOFAR_model",
    distribute_model=False,
    n_samples=1,
    timesteps=25
);

fig, ax = plotting.image_plots.plot_image_grid( samples[ 0 ] )
fig.savefig( "sample.png" );

#PyBDSF must read from a file, and only uses the PrimaryHDU
#so loop through all the files and save them to samples/
for i in range( samples.shape[ 0 ] ):
    hdul = fits.HDUList();
    hdu = fits.PrimaryHDU( samples[ i, -1, 0, :, : ] );

    #add CTYPE headers so PyBDSF doesn't complain
    #and dashes to conform to FITS standard
    hdu.header[ "CTYPE1" ] = "RA---   ";
    hdu.header[ "CTYPE2" ] = "DEC--   ";
    hdu.header[ "CDELT1" ] = 1.5; #arcseconds?
    hdu.header[ "CDELT2" ] = 1.5;


    hdul = fits.HDUList( [ hdu ] );
    hdul.writeto( "samples/sample" + str(i) + ".fits", "fix", overwrite=True );
    hdul.close();

    # have pybdsf read the image and save to samples/fit_samples/
    img = bdsf.process_image( 
        "samples/sample" + str(i) + ".fits", 
        beam = (0.033, 0.033, 0.0),
        thresh_isl = 5,
        thresh_pix = 0.5,
        mean_map = "const",
        rms_map = True,
        thresh = "hard",
        frequency = 144e6 );
    img.show_fit();
    img.export_image( img_type='gaus_model', outfile="samples/fit_samples/sample" + str(i) + ".fits", clobber=True );
    img.write_catalog( format='fits', catalog_type='srl', clobber=True );


    with fits.open( "samples/fit_samples/sample" + str(i) + ".fits" ) as hdul_model:
        model_image_data = hdul_model[ 0 ].data;
        print( model_image_data );

