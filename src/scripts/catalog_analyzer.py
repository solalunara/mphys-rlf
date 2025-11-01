import os;
import sys;
from astropy.io import fits;
import astropy.io;
import astropy.stats;
import astropy.io.fits
import numpy as np;
from pathlib import Path;
import matplotlib.pyplot as plt;


# Add the src directory to Python path so we can import modules
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_dir)

import scripts.fits_viewer;
from scripts.dataset_h5_to_fits import H5ToFitsConverter;
from scripts.image_analyzer import ImageAnalyzer;
from scripts.recursive_file_analyzer import RecursiveFileAnalyzer, HistogramErrorDrawer;

class CatalogAnalyzer( RecursiveFileAnalyzer ):
    """
    A class to interpret PyBDSF catalogs recursively under some root directory

    Parameters
    ----------
    path: Path | str
        The root directory to recursively search under
    """
    def FluxCounter( self ):
        """
        Recurse through the root dir to calculate the model fluxes

        Returns
        -------
        list[ tuple[ float, float ] ] | tuple[ float, float ]
            A list of the results of __FluxCounter() on all paths in the root dir, or the result of __FluxCounter() on the root if the root is a path to a file
        """
        return self.ForEach( CatalogAnalyzer.__FluxCounter, "fits" );

    def __FluxCounter( path: Path ):
        """
        Parameters
        ----------
        path: Path
            The path to the pybdsf log file

        Returns
        -------
        flux_total: float
            The flux of the MODEL (total of islands) in Jy or arbitrary units (because of 0-1 normalizaiton)
        e_flux_total: float
            The error on the flux of the MODEL (sums of gaussians convolved with the beam) in Jy or arbitrary units (because of 0-1 normalizaiton)
        """
        with fits.open( path ) as hdul:
            islands = hdul[ 1 ].data;
        flux_total = 0;
        e_flux_total = 0;
        for island in islands:
            flux_total += island[ 6 ];
            e_flux_total += np.sqrt( e_flux_total**2 + island[ 7 ]**2 );
        return flux_total, e_flux_total;


if __name__ == "__main__":

    #Create and analyze FITS images from the dataset if they don't exist
    if not Path( "fits_images/dataset" ).exists():
        converter = H5ToFitsConverter( "image_data/LOFAR/LOFAR_Dataset.h5", "fits_images/dataset" );
        converter.ConvertLOFAR( 10000, 1000, 100 );
    if not Path( "pybdsf_catalogs/dataset" ):
        dataset_analyzer = ImageAnalyzer( "dataset" );
        dataset_analyzer.AnalyzeAllFITSInInput();

    dataset_catalog_analyzer = CatalogAnalyzer( "pybdsf_catalogs/dataset/" );
    generated_catalog_analyzer = CatalogAnalyzer( "pybdsf_catalogs/generated/" );
    dataset_fluxes = np.array( dataset_catalog_analyzer.FluxCounter() ); #both fluxes and flux errors, (N,2)
    generated_fluxes = np.array( generated_catalog_analyzer.FluxCounter() ); #both fluxes and flux errors, (N,2)

    resolution = 1000;
    fig = plt.figure( figsize=(int(resolution*1/100), int(resolution/100)) );
    gs = fig.add_gridspec( 1, 1,
                            left=0.05, right=0.95, bottom=0.2, top=0.8,
                            wspace=0.5, hspace=0.5 );
    ax_flux = fig.add_subplot( gs[ 0, 0 ] );

    BINCOUNT = 10;
    hist = HistogramErrorDrawer();
    hist.Draw( dataset_fluxes[ :, 0 ], ax=ax_flux, bins=BINCOUNT, range=(0,30), label="dataset", color="b", density=True, log=True );
    hist.Draw( generated_fluxes[ :, 0 ], ax=ax_flux, bins=BINCOUNT, range=(0,30), label="generated", color="g", density=True, log=True );

    ax_flux.legend();
    ax_flux.set_title( "Model Fluxes" );

    plt.savefig( "hist.png" );