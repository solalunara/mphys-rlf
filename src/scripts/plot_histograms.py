import matplotlib.pyplot as plt
from pybdsf_analysis.recursive_file_analyzer import HistogramErrorDrawer
from pybdsf_analysis.image_analyzer import ImageAnalyzer
import argparse
import logging
import pybdsf_analysis.pybdsf_run_analysis
import utils.paths
from utils.distributed import DistributedUtils
import logging
from pybdsf_analysis.log_analyzer import LogAnalyzer
import pybdsf_analysis.log_analyzer as la
import pybdsf_analysis.recursive_file_analyzer as rfa
import numpy as np
from completeness.img_data_arrays import ImageDataArrays

def plot_graphs_with_pybdsf_data( log_level: int = logging.INFO ):
    resolution = 600
    BINCOUNT = 25
    NORM = True

    fig = plt.figure( figsize=(int(resolution/100), int(resolution/100)) )
    gs = fig.add_gridspec( 2, 2,
                            left=0.11, right=0.99, bottom=0.05, top=0.95,
                            wspace=0.25, hspace=0.25 )
    ax_flux = fig.add_subplot( gs[ 0, 0 ] )
    ax_mean = fig.add_subplot( gs[ 0, 1 ] )
    ax_rms = fig.add_subplot( gs[ 1, 0 ] )
    ax_pix = fig.add_subplot( gs[ 1, 1 ] )

    axes = [ ax_flux, ax_mean, ax_rms, ax_pix ]
    titles = [ "Flux", "Mean", "RMS", "Pixel Values" ]
    if NORM:
        xlabels = [ "Integrated Flux (arbitrary units)", "Image Mean (0-1)", "Image RMS (0-1)", "Pixel Value (0-1)" ]
        ranges = [ (0, 60), (0, 0.2), (0, 0.3), (0, 1) ]
    else:
        xlabels = [ "Integrated Flux (mJy)", "Image Mean (mJy/pix)", "Image RMS (mJy/pix)", "Pixel Value (mJy/pix)" ]
        ranges = [ (0, 10000), (0, 300), (0, 300), (0, 1000) ]
    ylabels = [ "Relative Frequency" ] * 4

    

    hist = HistogramErrorDrawer()
    for subdir in [ utils.paths.DATASET_SUBDIR, utils.paths.GENERATED_SUBDIR ]:
        data_arrays = ImageDataArrays( subdir )
        images = data_arrays.images
        model_fluxes = data_arrays.model_fluxes
        means = data_arrays.sigma_clipped_means
        rmsds = data_arrays.sigma_clipped_rmsds

        div_factor = data_arrays.image_scale_factors if NORM else np.array( [ 1 ] )
        axes_data = [ model_fluxes / div_factor, means / div_factor, rmsds / div_factor, ( images / div_factor[ :, np.newaxis, np.newaxis ] ).ravel() ]
        for ax, ax_data, range in zip( axes, axes_data, ranges ):
            hist.draw( ax_data,
                       ax=ax,
                       bins=BINCOUNT,
                       range=range,
                       label=subdir, 
                       color="b" if subdir == utils.paths.DATASET_SUBDIR else "g",
                       density=False,
                       relative=True )
                       

    for ax, title, range in zip( axes, titles, ranges ):
        ax.legend()
        ax.set_title( title )
        ax.set_yscale( 'log' )
        ax.set_xbound( lower=range )

    plt.savefig( "hist.png" )
    plt.show()


if __name__ == "__main__":
    pybdsf_analysis.pybdsf_run_analysis.analyze_everything()

    parser = argparse.ArgumentParser()
    parser.add_argument( "-v", "--verbose", help="Print a message to the console every time a file is read or a directory is entered", action='store_true' )
    args = parser.parse_args()
    verbose = args.verbose
    log_level = logging.DEBUG if verbose else logging.INFO

    du = DistributedUtils()
    du.single_task_only_last( 'plot_graphs_with_pybdsf_data', plot_graphs_with_pybdsf_data, 0, log_level )