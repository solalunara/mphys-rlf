from image_analyzer import ImageAnalyzer;
from pathlib import Path;
from astropy.io import fits;
from fits_viewer import FitsViewer;

if __name__ == "__main__":
    #get a generic residual image
    original_path = Path( "fits_images/dataset/50000-60000/image50042.fits" );
    gaus_model_path = Path( "fits_images/exported/dataset/gaus_model/50000-60000/image50042.fits" );

    with fits.open( original_path ) as original_file:
        original = original_file[ 0 ].data;
    with fits.open( gaus_model_path ) as gaus_model_file:
        gaus_model = gaus_model_file[ 0 ].data;

    gaus_resid = original - gaus_model;

    analyzer = ImageAnalyzer( "exported/dataset" );
    analyzer.SaveImageToFITS( gaus_resid, "gaus_resid/50000-59999/image50042.fits" );
    FitsViewer( "fits_images/exported/dataset/gaus_resid/50000-59999/image50042.fits" ).show_image_grid( "Gaus Resid 50042" );
