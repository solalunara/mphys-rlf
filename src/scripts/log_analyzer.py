import numpy as np;
from pathlib import Path;
import matplotlib.pyplot as plt;
import re;
from pybdsf_analysis.recursive_file_analyzer import RecursiveFileAnalyzer, HistogramErrorDrawer;
from pybdsf_analysis.image_analyzer import ImageAnalyzer;
import argparse;
import logging;
import scripts.pybdsf_run_analysis;
import utils.paths;

def FMR( path: Path ):
    """
    Parameters
    ----------
    path: Path
        The path to the pybdsf log file

    Returns
    -------
    flux: float
        The flux of the image in Jy or arbitrary units (because of 0-1 normalizaiton)
    mean: float
        The raw mean of the image in mJy or arbitrary units
    rms: float
        The raw rms of the image in mJy or arbitrary units
    """
    with open( str( path ) ) as file:
        filedata = file.read();
    #include re.DOTALL to make the .*? able to expand over newlines
    exp = re.compile( r"Raw mean \(Stokes I\) =  (\d+\.\d+) mJy and raw rms =  (\d+\.\d+) mJy.*?Flux from sum of \(non-blank\) pixels ..... : (\d+\.\d+) Jy", re.DOTALL );
    match = exp.search( filedata );
    mean = float( match.group( 1 ) );
    rms = float( match.group( 2 ) );
    flux = float( match.group( 3 ) );
    return flux, mean, rms;



if __name__ == "__main__":
    scripts.pybdsf_run_analysis.analyze_everything();

    parser = argparse.ArgumentParser();
    parser.add_argument( "-v", "--verbose", help="Print a message to the console every time a file is read or a directory is entered", action='store_true' );
    args = parser.parse_args();
    verbose = args.verbose;

    log_level = logging.DEBUG if verbose else logging.INFO;

    dataset_analyzer = ImageAnalyzer( utils.paths.DATASET_SUBDIR, log_level=log_level );
    generated_analyzer = ImageAnalyzer( utils.paths.GENERATED_SUBDIR, log_level=log_level );
    dataset_log_analyzer = RecursiveFileAnalyzer( utils.paths.PYBDSF_LOG_PARENT / utils.paths.DATASET_SUBDIR );
    generated_log_analyzer = RecursiveFileAnalyzer( utils.paths.PYBDSF_LOG_PARENT / utils.paths.GENERATED_SUBDIR );
    dataset_data = np.array( dataset_log_analyzer.ForEach( FMR, r'.*?\.log' ) );
    generated_data = np.array( generated_log_analyzer.ForEach( FMR, r'.*?\.log' ) );

    dataset_pix_vals = dataset_analyzer.GetPixelValues().ravel();
    generated_pix_vals = generated_analyzer.GetPixelValues().ravel();

    resolution = 1000;
    fig = plt.figure( figsize=(int(resolution*4/100), int(resolution/100)) );
    gs = fig.add_gridspec( 1, 4,
                            left=0.05, right=0.95, bottom=0.2, top=0.8,
                            wspace=0.5, hspace=0.5 );
    ax_flux = fig.add_subplot( gs[ 0, 0 ] );
    ax_mean = fig.add_subplot( gs[ 0, 1 ] );
    ax_rms = fig.add_subplot( gs[ 0, 2 ] );
    ax_pix = fig.add_subplot( gs[ 0, 3 ] );

    #Use numpy to get a histogram array of the values
    #and matplotlib to plot a log graph from the raw data
    hist = HistogramErrorDrawer();
    BINCOUNT = 100;
    hist.Draw( dataset_data[ :, 0 ], ax=ax_flux, bins=BINCOUNT, range=(0,50), label="dataset", color="b", density=True, log=True );
    hist.Draw( generated_data[ :, 0 ], ax=ax_flux, bins=BINCOUNT, range=(0,50), label="generated", color="g", density=True, log=True );
    hist.Draw( dataset_data[ :, 1 ], ax=ax_mean, bins=BINCOUNT, range=(0,300), label="dataset", color="b", density=True, log=True );
    hist.Draw( generated_data[ :, 1 ], ax=ax_mean, bins=BINCOUNT, range=(0,300), label="generated", color="g", density=True, log=True );
    hist.Draw( dataset_data[ :, 2 ], ax=ax_rms, bins=BINCOUNT, range=(0,500), label="dataset", color="b", density=True, log=True );
    hist.Draw( generated_data[ :, 2 ], ax=ax_rms, bins=BINCOUNT, range=(0,500), label="generated", color="g", density=True, log=True );
    hist.Draw( dataset_pix_vals, ax=ax_pix, bins=BINCOUNT, range=(0,1), label="dataset", color="b", density=True, log=True );
    hist.Draw( generated_pix_vals, ax=ax_pix, bins=BINCOUNT, range=(0,1), label="generated", color="g", density=True, log=True );

    ax_flux.legend();
    ax_mean.legend();
    ax_rms.legend();
    ax_pix.legend();

    ax_flux.set_title( "Flux" );
    ax_mean.set_title( "Mean" );
    ax_rms.set_title( "RMS" );
    ax_pix.set_title( "Pixel Values" );

    plt.savefig( "hist.png" );
