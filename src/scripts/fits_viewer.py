from astropy.io import fits;
import matplotlib.pyplot as plt;
import numpy as np;
from pathlib import Path;

class FitsViewer:
    """
    A utility class for viewing images in FITS files

    Parameters
    ----------
    file: Path | None = None
        The fits file to read
    """

    def __init__( self, file: Path | None = None ):
        if file is not None:
            self.read_from_file( file );
        else:
            self.data = np.empty( (0, 0) );

    def read_from_file( self, file: Path ):
        """
        Read fits file into data

        Parameters
        ----------
        file : Path
            The fits file to read the data from. The data should be stored in the PrimaryHDU, ideally
            in (n,n) shape but up to (1,1,...1,n,n) shape
        """
        with fits.open( str( file ) ) as hdul_model:
            self.data = hdul_model[ 0 ].data;
            #FITS files from gaus_model in pybdsf are of shape (1,1,n,n), so cut out all one shapes
            while self.data.shape[ 0 ] == 1:
                self.data = self.data[ 0 ];

    def show_image( self, 
                    title: str = 'Data', 
                    resolution: int = 80, 
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
        title : str = 'Data'
            Plot title
        resolution : int = 80
            Size of the full figure (independent of actual pixel size of the image)

        Gridspec Parameters
        -------------------
        left = 0.05,
        right = 0.95,
        bottom = 0.1,
        top = 0.95,
        wspace = 0.5,
        hspace = 0.5
        """
        if self.data.shape == (0, 0):
            raise RuntimeError( "Data has not been initialized. Please initialize data either with the constructor or read_from_file before calling show_image" );

        fig = plt.figure( figsize=(int(resolution/10), int(resolution/10)) );
        gs = fig.add_gridspec(1, 1,
                            left=left, right=right, bottom=bottom, top=top,
                            wspace=wspace, hspace=hspace);
        ax1 = fig.add_subplot( gs[ 0, 0 ] );
        ax1.set_title( title );
        img1 = ax1.imshow( self.data );
        img1.set_clim( 0, 1 );
        plt.show();
        return self;

    def read_and_show( self, 
                       file: Path, 
                       title: str = 'Data',
                       resolution: int = 80,
                       left: float = 0.05,
                       right: float = 0.95,
                       bottom: float = 0.1,
                       top: float = 0.95,
                       wspace: float = 0.5,
                       hspace: float = 0.5,
                        ):
        """
        Utility function to combine read_from_file and show_image

        Parameters
        ----------
        file : Path
            The fits file to read the data from. The data should be stored in the PrimaryHDU, ideally
            in (n,n) shape but up to (1,1,...1,n,n) shape
        title : str = 'Data'
            Plot title
        resolution : int = 80
            Size of the full figure (independent of actual pixel size of the image)

        Gridspec Parameters
        -------------------
        left = 0.05,
        right = 0.95,
        bottom = 0.1,
        top = 0.95,
        wspace = 0.5,
        hspace = 0.5
        """
        self.read_from_file( file );
        self.show_image( title, resolution, left, right, bottom, top, wspace, hspace );


if __name__ == "__main__":
    viewer = FitsViewer( "fits_images/exported/dataset/gaus_model/50000-60000/image50080.fits" ).show_image();

