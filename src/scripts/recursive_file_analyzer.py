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
from utils.logging import get_logger;

class RecursiveFileAnalyzer:
    def __init__( self, path: Path | str ):
        if path is not Path:
            path = Path( path );
        self.path = path;
        self.counter = 0;
        self.logger = get_logger( __name__ );
    
    def ForEach( self, function, ext: str | None = None, path: Path | str | None = None ):
        """
        A method to perform a generic function on all files within a directory given by path, or
        to read path as a file and perform the generic function on its contents, with optional file extension filtering

        Parameters
        ----------
        function : callable
            The function which will be called on each file in path recursively
        ext : str | None = None
            Only call the function and return values of the function on files that match this extension. If none match all files.
        path : Path | str | None = None
            The path to the file root, or any subdirectory or file. If none defaults to file root dir.

        Returns
        -------
        a list of the return values of the function on each file in path, or the return value acted on a particular file
        """
        if path is None:
            path = self.path;
            self.counter = 0;
        if path is not Path:
            path = Path( path );

        if path.is_dir():
            return_values = [];
            for sub_path in path.iterdir():
                sub_return_values = self.ForEach( function, ext, sub_path );

                #concatenate result lists so we end up with a big 1d array as a result
                #and toss None values (which we get if the extension doesn't match, or the fn itself returns None)
                # note - only files can return None, and files can only return one value, so we don't need None checking in array concat
                if isinstance( sub_return_values, list ):
                    return_values = return_values + sub_return_values;
                elif sub_return_values is not None:
                    return_values.append( sub_return_values );
            return return_values;
            
        else:
            if ( path.suffix == f".{ext}" ) or ( ext is None ):
                with open( str( path ), "r" ) as file:
                    return_value = function( file.read() );
                self.logger.debug( f"image log {self.counter}: {path}" );
                self.counter += 1;
                return return_value;
            else: return;