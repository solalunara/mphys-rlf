# This is a file created by Ashley and Luna. It defines the RecursiveFileAnalyzer and the HistogramErrorDrawer.
# The main purpose of the RecursiveFileAnalyzer is the method to get an unwrapped list of all the files in
# its directory recursively, as this is useful for multiprocessing or other scenareos where knowing the total
# number of files is helpful. The HistogramErrorDrawer is a utility class to house its Draw function, which
# draws a histogram and calculates its errors with astropy.stats.poisson_conf_interval

import astropy.stats;
import numpy as np;
from pathlib import Path;
import matplotlib.pyplot as plt;
import logging;
from utils.logging import get_logger;
from tqdm import tqdm;
import re;

class RecursiveFileAnalyzer:
    """
    A base class to act a function and return its value as a 1D list on all files (optionally matching an extension) under a given root directory

    If the OS environment variable 

    Parameters
    ----------
    path: Path | str
        The root directory to recursively search under
    log_level: int = logging.INFO
        The log level for the recursive file analyzer logger. When set to DEBUG, will log a message to the console when a directory is entered
        or a file read, with an index associated with each read file. Useful for slow operations to provide feedback on progress. Default logging.INFO
    """
    def __init__( self, path: Path | str, log_level: int = logging.INFO ):
        if path is not Path:
            path = Path( path );
        self.path = path;
        self.counter = 0;
        self.logger = get_logger( self.__class__.__name__ );
        self.logger.setLevel( log_level );

    def GetUnwrappedList( self, path: Path | None = None, pattern: str | None = None ):
        """
        Recurse through all files in path and unwrap all files into a single list,
        useful for multiprocessing

        Parameters
        ----------
        path: Path | None = None
            The path to unwrap. None defaults to root (self.path).
        pattern: str | None = None
            The regex pattern to search for. Items not matching will not be returned. If None, return all.

        Returns
        -------
        list[ Path ]
            An unwrapped list of all files in path, or the path itself if it is a fits file
        """
        if path is None:
            path = self.path;
        if path.is_dir():
            unwrapped_sublist = [];
            for iter_file in path.iterdir():
                result = self.GetUnwrappedList( iter_file, pattern );
                if isinstance( result, list ):
                    unwrapped_sublist = unwrapped_sublist + result;
                elif result is not None:
                    unwrapped_sublist.append( result );
            return unwrapped_sublist;
        elif ( pattern is None ) or ( re.match( pattern, str( path ) ) ):
            return path;
        return None;
    
    def ForEach( self, function, pattern: str | None = None ):
        """
        A method to perform a generic function on all files within a directory given by path, or
        to read path as a file and perform the generic function on its contents, with optional file extension filtering

        Parameters
        ----------
        function : callable
            The function which will be called on each file in path recursively
        pattern: str | None = None
            The regex pattern to search for. Items not matching will have the function operate on them. If None, operate on all.

        Returns
        -------
        list[ Any ]
            returns a list of the file return values within the path directory self.path
        """
        files = self.GetUnwrappedList( None, pattern );
        return_values = [];
        for file in tqdm( files, desc='Operating...', total=len( files ) ):
            return_values.append( function( file ) );
            self.logger.debug( f"Reading file {self.counter}: {file}" );
            self.counter += 1;
        return return_values;

class HistogramErrorDrawer:
    """
    Purely utility class to draw histograms with error bars
    """
    def __init__( self ):
        pass;

    def Draw( self, data: np.ndarray, ax: plt.Axes, bins: int, range: tuple[ float, float ], label: str, color: str, density: bool, log: bool ):
        """
        Utility function to draw a histogram with error bars according to astropy.stats.poisson_conf_interval with sigma=1.0

        Parameters
        ----------
        data: np.ndarray
            The data to plot
        ax: plt.Axes
            The axes to plot the histogram and error bars on
        bins: int
            Number of bins to sort the data into
        range: tuple[ float, float ]
            Range to plot the histogram on (neccesary parameter to compare histograms of slightly different data)
        label: str
            How to label the data
        color: str
            How to color the data
        density: bool
            Whether or not to make the histogram (and associated error bars) a density plot
        log: bool
            Whether or not to make the histogram (and associated error bars) a log plot
        """
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
