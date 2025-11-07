from astropy.io import fits;
import matplotlib.pyplot as plt;
import numpy as np;
from pathlib import Path;
import math;
import argparse;

class FitsViewer:
    """
    A utility class for viewing images in FITS files

    Parameters
    ----------
    *files: list[Path]
        The fits file(s) to read
    """

    def __init__( self, *files: list[Path] ):
        self.data = [];
        if len( files ) != 0:
            self.read_from_files( *files );

    def read_from_files( self, *files: list[Path] ):
        """
        Read fits file into data

        Parameters
        ----------
        *files : list[Path]
            The fits file to read the data from. The data should be stored in the PrimaryHDU, ideally
            in (n,n) shape but up to (1,1,...1,n,n) shape
        """
        for file in files:
            with fits.open( str( file ) ) as hdul:
                data = [ hdul[ i ].data for i in range( len( hdul ) ) ];

            #FITS files from gaus_model in pybdsf are of shape (1,1,n,n), so cut out all one shapes
            if len( data ) == 1:
                while data[ 0 ].shape[ 0 ] == 1:
                    data[ 0 ] = data[ 0 ][ 0 ];
            self.data.append( data );


    def show_image_grid( self, 
                         *titles: list[str|None],
                         resolution: int = 1000,
                         num_rows: int = 1,
                         outfile: str | None = None,
                         left = 0.05,
                         right = 0.95,
                         bottom = 0.1,
                         top = 0.95,
                         wspace = 0.5,
                         hspace = 0.5,
                          ):
        """
        Show a plot image for the data, and return this object for syntax sugar

        Parameters
        ----------
        *titles : list[str|None]
            Plot titles list of the same length as files, where None indicates to not set a title for the image
        resolution : int = 1000
            Size of the full figure (independent of actual pixel size of the image)
        num_rows : int = 1
            Number of rows to display the fits images using
        outfile : str | None = None
            Where to write the output figure to, or none to not write one

        Gridspec Parameters
        -------------------
        left = 0.05,
        right = 0.95,
        bottom = 0.1,
        top = 0.95,
        wspace = 0.5,
        hspace = 0.5
        """
        if self.data is None:
            raise RuntimeError( "Data has not been initialized. Please initialize data either with the constructor or read_from_file before calling show_image" );
        if len( titles ) != len( self.data ):
            raise RuntimeError( "Length of titles and length of data mismatch!" );

        fig = plt.figure( figsize=(resolution/100, resolution/100) );
        num_cols = int( math.ceil( len( titles ) / num_rows ) );
        gs = fig.add_gridspec( num_rows, num_cols,
                               left=left, right=right, bottom=bottom, top=top,
                               wspace=wspace, hspace=hspace );
        axes = [];
        for i in range( len( titles ) ):
            if len( self.data[ i ] ) != 1:
                raise RuntimeError( "All images must be stored as one image per file in the PrimaryHDU for compatibility with PyBDSF" );
            col = int( np.round( i / num_rows ) );
            row = i % num_rows;
            axes.append( fig.add_subplot( gs[ row, col ] ) );
            if titles[ i ] is not None:
                axes[ -1 ].set_title( titles[ i ] );
            img = axes[ -1 ].imshow( self.data[ i ][ 0 ] );
            img.set_clim( 0, 1 );
        if outfile is not None:
            plt.savefig( outfile )
        plt.show();
        return self;

if __name__ == "__main__":
    # Display any fits files passed as arguments
    parser = argparse.ArgumentParser( prog='python fits_viewer.py', 
                                      usage='%(prog)s [-h|--help] [--rows ROWS] [--resolution RESOLUTION] [-o|--outfile OUTFILE] [--left|--right|--top|--bottom|--wspace|--hspace VALUE] [FILES]',
                                      description='A program to visualize FITS images passed as arguments' );
    parser.add_argument( "--rows", help="How many rows to display the fits images in, default 1", type=int, default=1 );
    parser.add_argument( "--resolution", help="Resolution to display the image at, default 1000", type=int, default=1000 );
    parser.add_argument( "-o", "--outfile", help="Where to write the output file to. By default does not write figure to a file.", type=str, default=None );
    parser.add_argument( "--left", help="Gridspec left parameter default 0.05", type=float, default=0.05 );
    parser.add_argument( "--right", help="Gridspec right parameter default 0.95", type=float, default=0.95 );
    parser.add_argument( "--top", help="Gridspec top parameter default 0.95", type=float, default=0.95 );
    parser.add_argument( "--bottom", help="Gridspec bottom parameter default 0.1", type=float, default=0.1 );
    parser.add_argument( "--wspace", help="Gridspec wspace parameter default 0.5", type=float, default=0.5 );
    parser.add_argument( "--hspace", help="Gridspec hspace parameter default 0.5", type=float, default=0.5 );
    parser.add_argument( "FILES", nargs='*' );
    args = parser.parse_args();

    if len( args.FILES ) > 0:
        paths: list[ Path ] = [];
        titles: list[ str ] = [];
        for element in args.FILES:
            paths.append( Path( element ) );
            titles.append( paths[ -1 ].name );

        viewer = FitsViewer( *paths ).show_image_grid( *titles, 
                                                       resolution=args.resolution, 
                                                       num_rows=args.rows, 
                                                       outfile=args.outfile,
                                                       left=args.left,
                                                       right=args.right,
                                                       bottom=args.bottom,
                                                       top=args.top,
                                                       wspace=args.wspace,
                                                       hspace=args.hspace );

