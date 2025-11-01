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
from scripts.recursive_file_analyzer import RecursiveFileAnalyzer;

class LogAnalyzer( RecursiveFileAnalyzer ):
    def FluxCounter( self ):
        return self.ForEach( LogAnalyzer.CalculateFileFlux, "log" );

    def CalculateFileFlux( filedata: str ):
        line = "Flux from sum of (non-blank) pixels ..... : ";
        index = filedata.index( line );
        index += len( line );
        endindex = filedata.index( "Jy", index ) - 1;
        flux = float( filedata[ index:endindex ] );
        return flux;



if __name__ == "__main__":

    #Create and analyze FITS images from the dataset if they don't exist
    if not Path( "fits_images/dataset" ).exists():
        converter = H5ToFitsConverter( "image_data/LOFAR/LOFAR_Dataset.h5", "fits_images/dataset" );
        converter.ConvertLOFAR( 10000, 1000, 100 );
    if not Path( "pybdsf_catalogs/dataset" ):
        dataset_analyzer = ImageAnalyzer( "dataset" );
        dataset_analyzer.AnalyzeAllFITSInInput();

    dataset_catalog_analyzer = LogAnalyzer( "fits_images/dataset/" );
    generated_catalog_analyzer = LogAnalyzer( "fits_images/generated/" );
    dataset_fluxes = np.array( dataset_catalog_analyzer.FluxCounter() );
    generated_fluxes = np.array( generated_catalog_analyzer.FluxCounter() );

    #Use numpy to get a histogram array of the values
    #and matplotlib to plto a log graph from the raw data
    BINCOUNT = 10;
    d_hist, _ = np.histogram( dataset_fluxes, bins=BINCOUNT, range=(0,30) );
    g_hist, _ = np.histogram( generated_fluxes, bins=BINCOUNT, range=(0,30) );
    d_log_hist, bins, _ = plt.hist( dataset_fluxes, density=True, log=True, histtype='step', bins=BINCOUNT, label="dataset", color="b", range=(0,30) );
    g_log_hist, bins, _ = plt.hist( generated_fluxes, density=True, log=True, histtype='step', bins=BINCOUNT, label="generated", color="g", range=(0,30) );

    #Put errorbars on the centre of each bin using the poisson confidence interval
    bin_width = bins[ 1 ] - bins[ 0 ];
    bin_centres = bins[ :-1 ] + bin_width/2.0;
    d_conf_interval = astropy.stats.poisson_conf_interval( d_hist, sigma=1.0 );
    g_conf_interval = astropy.stats.poisson_conf_interval( g_hist, sigma=1.0 );
    d_conf_interval = np.where( d_conf_interval > 0, d_conf_interval, 1e-10 );
    g_conf_interval = np.where( g_conf_interval > 0, g_conf_interval, 1e-10 );
    d_yerr = np.log10( d_conf_interval[ 1 ] / d_conf_interval[ 0 ] ) / np.sum( d_hist ); #acounting for density weighting
    g_yerr = np.log10( g_conf_interval[ 1 ] / g_conf_interval[ 0 ] ) / np.sum( g_hist ); #acounting for density weighting

    plt.errorbar( bin_centres, d_log_hist, d_yerr, fmt='.', color='b' );
    plt.errorbar( bin_centres, g_log_hist, g_yerr, fmt='.', color='g' );
    plt.legend();
    plt.savefig( "hist.png" );