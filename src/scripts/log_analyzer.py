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
from scripts.recursive_file_analyzer import RecursiveFileAnalyzer, HistogramErrorDrawer;

class LogAnalyzer( RecursiveFileAnalyzer ):
    """
    A class to interpret PyBDSF logs recursively under some root directory

    Parameters
    ----------
    path: Path | str
        The root directory to recursively search under
    """
    def FMR( self ):
        """
        It's computationally cheaper to recurse through all log files once and do one regex match to each of them

        Returns
        -------
        list[ tuple[ float, float, float ] ] | tuple[ float, float, float ]
            A list of the results of __FMR() on all paths in the root dir, or the result of __FMR() on the root if the root is a path to a file
        """
        return self.ForEach( LogAnalyzer.__FMR, "log" );

    def __FMR( path: Path ):
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

    #Create and analyze FITS images from the dataset if they don't exist
    if not Path( "fits_images/dataset" ).exists():
        converter = H5ToFitsConverter( "image_data/LOFAR/LOFAR_Dataset.h5", "fits_images/dataset" );
        converter.ConvertLOFAR( 10000, 1000, 100 );
    if not Path( "pybdsf_catalogs/dataset" ):
        dataset_analyzer = ImageAnalyzer( "dataset" );
        dataset_analyzer.AnalyzeAllFITSInInput();

    dataset_catalog_analyzer = LogAnalyzer( "fits_images/dataset/" );
    generated_catalog_analyzer = LogAnalyzer( "fits_images/generated/" );
    dataset_data = np.array( dataset_catalog_analyzer.FMR() );
    generated_data = np.array( generated_catalog_analyzer.FMR() );

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
    hist = HistogramErrorDrawer();
    BINCOUNT = 100;
    hist.Draw( dataset_data[ :, 0 ], ax=ax_flux, bins=BINCOUNT, range=(0,50), label="dataset", color="b", density=True, log=True );
    hist.Draw( generated_data[ :, 0 ], ax=ax_flux, bins=BINCOUNT, range=(0,50), label="generated", color="g", density=True, log=True );
    hist.Draw( dataset_data[ :, 1 ], ax=ax_mean, bins=BINCOUNT, range=(0,300), label="dataset", color="b", density=True, log=True );
    hist.Draw( generated_data[ :, 1 ], ax=ax_mean, bins=BINCOUNT, range=(0,300), label="generated", color="g", density=True, log=True );
    hist.Draw( dataset_data[ :, 2 ], ax=ax_rms, bins=BINCOUNT, range=(0,500), label="dataset", color="b", density=True, log=True );
    hist.Draw( generated_data[ :, 2 ], ax=ax_rms, bins=BINCOUNT, range=(0,500), label="generated", color="g", density=True, log=True );

    ax_flux.legend();
    ax_mean.legend();
    ax_rms.legend();

    ax_flux.set_title( "Flux" );
    ax_mean.set_title( "Mean" );
    ax_rms.set_title( "RMS" );

    plt.savefig( "hist.png" );
