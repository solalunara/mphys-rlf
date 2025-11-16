import matplotlib.pyplot as plt;
from pybdsf_analysis.recursive_file_analyzer import HistogramErrorDrawer;
from pybdsf_analysis.image_analyzer import ImageAnalyzer;
import argparse;
import logging;
import pybdsf_analysis.pybdsf_run_analysis;
import utils.paths;
from utils.distributed import DistributedUtils;
import logging;
from pybdsf_analysis.log_analyzer import LogAnalyzer;
import numpy as np;

def plot_graphs_with_pybdsf_data( log_level: int = logging.INFO ):
    resolution = 1000;
    fig = plt.figure( figsize=(int(resolution*4/100), int(resolution/100)) );
    gs = fig.add_gridspec( 1, 4,
                            left=0.05, right=0.95, bottom=0.2, top=0.8,
                            wspace=0.5, hspace=0.5 );
    ax_flux = fig.add_subplot( gs[ 0, 0 ] );
    ax_mean = fig.add_subplot( gs[ 0, 1 ] );
    ax_rms = fig.add_subplot( gs[ 0, 2 ] );
    ax_pix = fig.add_subplot( gs[ 0, 3 ] );

    dataset_logs = LogAnalyzer( utils.paths.DATASET_SUBDIR );
    generated_logs = LogAnalyzer( utils.paths.GENERATED_SUBDIR );

    dataset_data = np.array( dataset_logs.fmr_data() );
    generated_data = np.array( generated_logs.fmr_data() );

    dataset_analyzer = ImageAnalyzer( utils.paths.DATASET_SUBDIR, log_level=log_level );
    dataset_pix_vals = dataset_analyzer.get_pixel_values().ravel();
    generated_analyzer = ImageAnalyzer( utils.paths.GENERATED_SUBDIR, log_level=log_level );
    generated_pix_vals = generated_analyzer.get_pixel_values().ravel();

    #Use numpy to get a histogram array of the values
    #and matplotlib to plot a log graph from the raw data
    hist = HistogramErrorDrawer();
    BINCOUNT = 100;
    hist.draw( dataset_data[ :, 0 ], ax=ax_flux, bins=BINCOUNT, range=(0,50), label="dataset", color="b", density=True, log=True );
    hist.draw( generated_data[ :, 0 ], ax=ax_flux, bins=BINCOUNT, range=(0,50), label="generated", color="g", density=True, log=True );
    hist.draw( dataset_data[ :, 1 ], ax=ax_mean, bins=BINCOUNT, range=(0,300), label="dataset", color="b", density=True, log=True );
    hist.draw( generated_data[ :, 1 ], ax=ax_mean, bins=BINCOUNT, range=(0,300), label="generated", color="g", density=True, log=True );
    hist.draw( dataset_data[ :, 2 ], ax=ax_rms, bins=BINCOUNT, range=(0,500), label="dataset", color="b", density=True, log=True );
    hist.draw( generated_data[ :, 2 ], ax=ax_rms, bins=BINCOUNT, range=(0,500), label="generated", color="g", density=True, log=True );
    hist.draw( dataset_pix_vals, ax=ax_pix, bins=BINCOUNT, range=(0,1), label="dataset", color="b", density=True, log=True );
    hist.draw( generated_pix_vals, ax=ax_pix, bins=BINCOUNT, range=(0,1), label="generated", color="g", density=True, log=True );

    ax_flux.legend();
    ax_mean.legend();
    ax_rms.legend();
    ax_pix.legend();

    ax_flux.set_title( "Flux" );
    ax_mean.set_title( "Mean" );
    ax_rms.set_title( "RMS" );
    ax_pix.set_title( "Pixel Values" );

    plt.savefig( "hist.png" );


if __name__ == "__main__":
    pybdsf_analysis.pybdsf_run_analysis.analyze_everything();

    parser = argparse.ArgumentParser();
    parser.add_argument( "-v", "--verbose", help="Print a message to the console every time a file is read or a directory is entered", action='store_true' );
    args = parser.parse_args();
    verbose = args.verbose;
    log_level = logging.DEBUG if verbose else logging.INFO;

    du = DistributedUtils();
    du.last_task_only( 'plot_graphs_with_pybdsf_data', plot_graphs_with_pybdsf_data, log_level )