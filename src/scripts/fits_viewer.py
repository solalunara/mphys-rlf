from astropy.io import fits;
import matplotlib.pyplot as plt;
import numpy as np;
from pathlib import Path;
import math;

class FitsViewer:
    """
    A utility class for viewing images in FITS files

    Parameters
    ----------
    *files: list[Path]
        The fits file(s) to read
    """

    def __init__( self, *files: list[Path] ):
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
        self.data = None;
        for file in files:
            with fits.open( str( file ) ) as hdul_model:
                data = hdul_model[ 0 ].data;

            #FITS files from gaus_model in pybdsf are of shape (1,1,n,n), so cut out all one shapes
            while data.shape[ 0 ] == 1:
                data = data[ 0 ];
            data = data[ np.newaxis, :, : ];
            if self.data is None:
                self.data = data;
            else:
                self.data = np.concatenate( (self.data, data), axis=0 );


    def show_image_grid( self, 
                         *titles: list[str],
                         resolution: int = 1000,
                         num_rows: int = 1,
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
        resolution : int = 1000
            Size of the full figure (independent of actual pixel size of the image)
        num_rows : int = 1
            Number of rows to display the fits images using
        *titles : list[str]
            Plot titles of the same length as files

        Gridspec Parameters
        -------------------
        left = 0.05,
        right = 0.95,
        bottom = 0.1,
        top = 0.95,
        wspace = 0.5,
        hspace = 0.5
        """
        if self.data.shape == (0, 0, 0, 0):
            raise RuntimeError( "Data has not been initialized. Please initialize data either with the constructor or read_from_file before calling show_image" );
        if len( titles ) != self.data.shape[ 0 ]:
            raise RuntimeError( "Length of titles and length of data mismatch!" );

        fig = plt.figure( figsize=(int(resolution/100), int(resolution/100)) );
        num_cols = int( math.ceil( len( titles ) / num_rows ) );
        gs = fig.add_gridspec( num_rows, num_cols,
                               left=left, right=right, bottom=bottom, top=top,
                               wspace=wspace, hspace=hspace );
        axes = [];
        for i in range( len( titles ) ):
            col = int( np.round( i / num_rows ) );
            row = i % num_rows;
            axes.append( fig.add_subplot( gs[ row, col ] ) );
            axes[ -1 ].set_title( titles[ i ] );
            img = axes[ -1 ].imshow( self.data[ i ] );
            img.set_clim( 0, 1 );
        plt.show();
        return self;

if __name__ == "__main__":
    viewer = FitsViewer( "fits_images/dataset/50000-60000/image50080.fits",
                         "fits_images/dataset/50000-60000/image50081.fits",
                         "fits_images/dataset/50000-60000/image50082.fits",
                         ).show_image_grid( "image 50080",
                                            "image 50081",
                                            "image 50082" );

