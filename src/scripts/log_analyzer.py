import os;
import sys;
from astropy.io import fits;
import astropy.io;
import astropy.stats;
import astropy.io.fits
import numpy as np;
from pathlib import Path;
import matplotlib.pyplot as plt;
import re;


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

    def MeanAndRMS( self ):
        return self.ForEach( LogAnalyzer.GetMeanAndRMS, "log" );

    def CalculateFileFlux( filedata: str ):
        line = "Flux from sum of (non-blank) pixels ..... : ";
        index = filedata.index( line );
        index += len( line );
        endindex = filedata.index( "Jy", index ) - 1;
        flux = float( filedata[ index:endindex ] );
        return flux;

    def GetMeanAndRMS( filedata: str ):
        exp = re.compile( r"Raw mean \(Stokes I\) =  (\d+\.\d+) mJy and raw rms =  (\d+\.\d+) mJy" );
        match = exp.search( filedata );
        mean = float( match.group( 1 ) );
        rms = float( match.group( 2 ) );
        return mean, rms;


def DrawHistogramWithErrorbars( data: np.ndarray, ax: plt.Axes, bins: int, range: tuple[ float, float ], label: str, color: str, density: bool, log: bool ):
    hist, _ = np.histogram( data, bins=bins, range=range );
    drawn_histogram, bin_data, _ = ax.hist( data, density=density, log=log, histtype='step', bins=bins, label=label, color=color, range=range );
    bin_width = bin_data[ 1 ] - bin_data[ 0 ];
    bin_centres = bin_data[ :-1 ] + bin_width/2.0;
    conf_interval = astropy.stats.poisson_conf_interval( hist, sigma=1.0 );

    yerr = 0;
    if log:
        conf_interval = np.where( conf_interval > 0, conf_interval, 1e-10 ); #zeroes cause errors when log=True
        yerr = np.log10( conf_interval[ 1 ] / conf_interval[ 0 ] );
    else:
        yerr = conf_interval[ 1 ] - conf_interval[ 0 ];

    #poisson_conf_interval needs the raw data to be accurate, so we do it on the unweighted histogram and weight it afterward here
    if density:
        yerr /= np.sum( data );

    ax.errorbar( bin_centres, drawn_histogram, yerr, fmt='.', color=color );



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

    dataset_mean_and_rms = np.array( dataset_catalog_analyzer.MeanAndRMS() );
    generated_mean_and_rms = np.array( generated_catalog_analyzer.MeanAndRMS() );

    resolution = 1000;
    fig = plt.figure( figsize=(int(resolution*3/100), int(resolution/100)) );
    gs = fig.add_gridspec( 1, 3,
                            left=0.05, right=0.95, bottom=0.2, top=0.8,
                            wspace=0.5, hspace=0.5 );
    ax_flux = fig.add_subplot( gs[ 0, 0 ] );
    ax_mean = fig.add_subplot( gs[ 0, 1 ] );
    ax_rms = fig.add_subplot( gs[ 0, 2 ] );

    #Use numpy to get a histogram array of the values
    #and matplotlib to plot a log graph from the raw data
    BINCOUNT = 30;
    DrawHistogramWithErrorbars( dataset_fluxes, ax=ax_flux, bins=BINCOUNT, range=(0,30), label="dataset", color="b", density=True, log=True );
    DrawHistogramWithErrorbars( generated_fluxes, ax=ax_flux, bins=BINCOUNT, range=(0,30), label="generated", color="g", density=True, log=True );
    DrawHistogramWithErrorbars( dataset_mean_and_rms[ :, 0 ], ax=ax_mean, bins=BINCOUNT, range=(0,150), label="dataset", color="b", density=True, log=True );
    DrawHistogramWithErrorbars( generated_mean_and_rms[ :, 0 ], ax=ax_mean, bins=BINCOUNT, range=(0,150), label="generated", color="g", density=True, log=True );
    DrawHistogramWithErrorbars( dataset_mean_and_rms[ :, 1 ], ax=ax_rms, bins=BINCOUNT, range=(0,200), label="dataset", color="b", density=True, log=True );
    DrawHistogramWithErrorbars( generated_mean_and_rms[ :, 1 ], ax=ax_rms, bins=BINCOUNT, range=(0,200), label="generated", color="g", density=True, log=True );

    ax_flux.legend();
    ax_mean.legend();
    ax_rms.legend();

    ax_flux.set_title( "Flux" );
    ax_mean.set_title( "Mean" );
    ax_rms.set_title( "RMS" );

    plt.savefig( "hist.png" );