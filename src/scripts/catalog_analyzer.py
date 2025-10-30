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

class CatalogAnalyzer:
    def __init__( self, path: Path | str ):
        if path is not Path:
            path = Path( path );
        self.path = path;
        self.counter = 0;
    
    def ForEach( self, function, path: Path | str | None = None ):
        """
        A method to perform a generic function on all hdul's within a directory given by path, or
        to read path as a file and perform the generic function on its hdul, assuming it is a fits file.

        Parameters
        ----------
        function : callable
            The function which will be called on each hdul in path recursively
        path : Path | None
            The path to the catalog root, or any subdirectory or catalog. If none defaults to catalog root dir.

        Returns
        -------
        a list of the return values of the function on each catalog in path, or the return value acted on a particular catalog
        """
        if path is None:
            path = self.path;
            self.counter = 0;
        if path is not Path:
            path = Path( path );

        if path.is_dir():
            return_values = [];
            for sub_path in path.iterdir():
                sub_return_values = self.ForEach( function, sub_path );
                if isinstance( sub_return_values, list ):
                    return_values = return_values + sub_return_values;
                else:
                    return_values.append( sub_return_values );
            return return_values;
            
        else:
            if path.suffix == ".fits":
                with fits.open( str( path ) ) as hdul:
                    return_value = function( hdul );
                print( f"image {self.counter}: {path}")
                self.counter += 1;
                return return_value;
            else: return;

    def FluxCounter( self ):
        return self.ForEach( CalculateHDULFlux );

def CalculateHDULFlux( hdul: astropy.io.fits.hdu.HDUList ):
    islands = hdul[ 1 ].data;
    flux_total = 0;
    e_flux_total = 0;
    for island in islands:
        flux_total += island[ 6 ];
        e_flux_total += np.sqrt( e_flux_total**2 + island[ 7 ]**2 );
    return flux_total, e_flux_total;


if __name__ == "__main__":
    catalog_analyzer = CatalogAnalyzer( Path( "pybdsf_catalogs/dataset/" ) );
    fluxes = np.array( catalog_analyzer.FluxCounter() ); #both fluxes and flux errors, (N,2)

    #Use numpy to get a histogram array of the values
    #and matplotlib to plto a log graph from the raw data
    BINCOUNT = 10;
    hist, _ = np.histogram( fluxes[ :, 0 ], bins=BINCOUNT );
    log_hist, bins, _ = plt.hist( fluxes[ :, 0 ], density=True, log=True, histtype='step', bins=BINCOUNT );

    #Put errorbars on the centre of each bin using the poisson confidence interval
    bin_width = bins[ 1 ] - bins[ 0 ];
    bin_centres = bins[ :-1 ] + bin_width/2.0;
    conf_interval = astropy.stats.poisson_conf_interval( hist, sigma=1.0 );
    conf_interval = np.where( conf_interval > 0, conf_interval, 1e-10 );
    yerr = np.log10( conf_interval[ 1 ] / conf_interval[ 0 ] ) / np.sum( hist ); #acounting for density weighting

    plt.errorbar( bin_centres, log_hist, yerr, fmt='.' );
    plt.savefig( "hist.png" );