from astropy.io import fits;
import numpy as np;
from pathlib import Path;
import matplotlib.pyplot as plt;
import utils.paths;
from pybdsf_analysis.recursive_file_analyzer import RecursiveFileAnalyzer, HistogramErrorDrawer;
import argparse;
import logging;
import analyze_gen_and_data; #This will do all dataset prep automatically

def FluxCounter( path: Path ):
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
    parser = argparse.ArgumentParser();
    parser.add_argument( "-v", "--verbose", help="Print a message to the console every time a file is read or a directory is entered", action='store_true' );
    args = parser.parse_args();
    verbose = args.verbose;

    log_level = logging.DEBUG if verbose else logging.INFO;

    dataset_catalog_analyzer = RecursiveFileAnalyzer( utils.paths.PYBDSF_ANALYSIS_PARENT / utils.paths.DATASET_SUBDIR, log_level );
    generated_catalog_analyzer = RecursiveFileAnalyzer( utils.paths.PYBDSF_ANALYSIS_PARENT / utils.paths.GENERATED_SUBDIR, log_level );
    dataset_fluxes = np.array( dataset_catalog_analyzer.ForEach( FluxCounter, 'fits' ) ); #both fluxes and flux errors, (N,2)
    generated_fluxes = np.array( generated_catalog_analyzer.ForEach( FluxCounter, 'fits' ) ); #both fluxes and flux errors, (N,2)

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