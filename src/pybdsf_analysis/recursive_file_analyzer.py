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
        self.logger = get_logger( self.__class__.__name__ );
        self.logger.setLevel( log_level );

    def get_unwrapped_list( self, path: Path | None = None, pattern: str | None = None, numeric_range: tuple[int,int] | None = None, return_nums: bool = False ):
        """
        Recurse through all files in path and unwrap all files into a single list,
        useful for multiprocessing

        Parameters
        ----------
        path: Path | None = None
            The path to unwrap. None defaults to root (self.path).
        pattern: str | None = None
            The regex pattern to search for. Items not matching will not be returned. If None, return all.
        numeric_range: int | None = None
            If there is a regex pattern to search for and it has a capture group, attempt to parse the capture
            group into an integer. If the integer is within the numeric range (inclusive begin, exclusive end), 
            match, otherwise don't match. None matches all.
        return_nums: bool = False
            If there is a regex pattern to search for and it has a capture group, attempt to parse the capture
            group into an integer. If return_nums is true, this will be returned with the path such that the return
            type of this function is list[ (Path, int) ]

        Returns
        -------
        list[ Path ] or list[ (Path, int) ]
            An unwrapped list of all files in path, or the path itself if it is a fits file
        """
        if path is None:
            path = self.path;
        if path.is_dir():
            unwrapped_sublist = [];
            for iter_file in path.iterdir():
                result = self.get_unwrapped_list( iter_file, pattern, numeric_range, return_nums );
                if isinstance( result, list ):
                    unwrapped_sublist = unwrapped_sublist + result;
                elif result is not None:
                    unwrapped_sublist.append( result );
            return unwrapped_sublist;

        # Check number is in numeric range if passed
        elif ( pattern is None ) or ( re.match( pattern, str( path ) ) ):
            return_value = path;
            if ( numeric_range is not None ) or ( return_nums ):
                try:
                    number_str = re.search( pattern, str( path ) ).group( 1 );
                    number = int( number_str );
                    if numeric_range is not None:
                        if ( number >= numeric_range[ 1 ] ) or ( number < numeric_range[ 0 ] ):
                            return None;
                    if return_nums:
                        return_value = ( path, number );
                except IndexError:
                    if numeric_range is not None:
                        self.logger.warning( f'Numeric range ({numeric_range[ 0 ]},{numeric_range[ 1 ]}) provided but pattern {pattern} has no capture group' );
                    else:
                        self.logger.warning( f'Tried to return numbers for each file provided but pattern {pattern} has no capture group' );
                except ValueError:
                    self.logger.error( f'Captured {number_str} cannot be converted to an integer' );
            return return_value;
        return None;
    
    def for_each( self, function, path: Path | None = None, pattern: str | None = None, progress_bar: bool = False, numeric_range: tuple[int,int] | None = None, return_nums: bool = False ):
        """
        A method to perform a generic function on all files within a directory given by path, or
        to read path as a file and perform the generic function on its contents, with optional file extension filtering

        Parameters
        ----------
        function : callable
            The function which will be called on each file in path recursively
        path : Path | None = None
            The path to do the for_each on, or None to do the for_each on self.path
        pattern : str | None = None
            The regex pattern to search for. Items not matching will have the function operate on them. If None, operate on all.
        progress_bar : bool = False
            Whether or not to show a progress bar for the function operating on each file
        numeric_range: int | None = None
            If there is a regex pattern to search for and it has a capture group, attempt to parse the capture
            group into an integer. If the integer is within the numeric range (inclusive begin, exclusive end), 
            match, otherwise don't match. None matches all.
        return_nums: bool = False
            If there is a regex pattern to search for and it has a capture group, attempt to parse the capture
            group into an integer. If return_nums is true, this will be returned with the path such that the return
            type of this function is of shape ( len( files ), 2 )

        Returns
        -------
        np.ndarray
            returns a list of the file return values within the path directory self.path, of shape ( len( files ) ) if
            return_nums is False else ( len( files ), 2 )
        """
        files = self.get_unwrapped_list( path, pattern, numeric_range, return_nums );
        return_values = [ None ] * len( files );
        if return_nums:
            return_numbers = np.empty( (len( files )) );
        i = 0;
        if progress_bar:
            arr = tqdm( files, desc='Operating...', total=len( files ) );
        else:
            arr = files;
        for file in arr:
            if return_nums:
                return_values[ i ] = function( file[ 0 ] );
                return_numbers[ i ] = file[ 1 ];
                self.logger.debug( f"Reading file {file[ 1 ]}: {file[ 0 ]}" );
            else:
                return_values[ i ] = function( file );
                self.logger.debug( f"Reading file {file}" );
            i += 1;

        if return_nums:
            return return_values, return_numbers;
        else:
            return return_values;

class HistogramErrorDrawer:
    """
    Purely utility class to draw histograms with error bars
    """
    def __init__( self ):
        pass;

    def draw( self, data: np.ndarray, ax: plt.Axes, bins: int, range: tuple[ float, float ], label: str, color: str, density: bool, log: bool ):
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
