from utils.logging import get_logger;
from files.dataset import LOFAR_DATA_PATH, single_node_download_dataset;
from utils.distributed import DistributedUtils;
import utils.paths as pth;
import h5py;
import numpy as np;
from pathlib import Path;
from sklearn.preprocessing import PowerTransformer;
from files.paths import single_node_prepare_folders;

def write_maxvals_of_h5_to_file( outfile: Path, infile: Path ):
    """
    A function to go through an input h5 file, select the 'images' category, and sum along axes 1 and 2 to get an array of the maximum
    pixel values of each image, then save the numpy array to an output file. This saves the maximum pixel values neccesary for the box-cox
    power transform in a much more portable format, shrinking the file size by 6400x and allowing it to be copied for access by multiple nodes.

    Parameters
    ----------
    outfile : Path
        The output to write the numpy max vals array to
    infile : Path
        The h5 file which contains images as file[ 'images' ][ : ].shape = (n_images, ndim, ndim)
    """
    with h5py.File( infile, "r" ) as f:
        max_vals = np.max( f[ "images" ][ : ], axis=(1, 2) );
    np.save( outfile, max_vals );

class PeakFluxPowerTransformer:
    """
    A utility class to easily power transform peak flux values based on the max values in the dataset, without
    having to constantly validate files exist
    """
    def __init__( self ):

        # Get a distribution of scaled max fluxes from the lofar data
        # This requires:
        #   1/ The dataset is downloaded
        #   2/ The max values of the dataset to be saved to a file using only one node while the rest wait,
        #   3/ Those max values are used to fit a box-cox transform
        self.logger = get_logger( __name__ );
        self.du = DistributedUtils();

        self.logger.debug( 'Making sure we have the dataset...' );
        if not LOFAR_DATA_PATH.exists():
            single_node_download_dataset();

        self.logger.debug( 'Making sure all folders exist...' );        
        single_node_prepare_folders();

        self.maxvals_path = pth.NP_ARRAY_PARENT / 'maxvals.npy';
        self.logger.debug( 'Dataset downloaded. Saving dataset maxvals to %s...', self.maxvals_path );
        if not self.maxvals_path.exists():
            self.du.single_task_only_forcewait( 'write_maxvals_of_h5_to_file', write_maxvals_of_h5_to_file, 0, self.maxvals_path, LOFAR_DATA_PATH );
        self.logger.debug( 'Done with shared file IO' );

        self.pt = PowerTransformer( method="box-cox" );
        self.pt.fit( np.load( self.maxvals_path ).reshape(-1, 1) );

    def transform( self, array: np.ndarray ):
        return self.pt.transform( array.reshape( -1, 1 ) )[ :, 0 ];

    def inverse_transform( self, array: np.ndarray ):
        return self.pt.inverse_transform( array.reshape( -1, 1 ) )[ :, 0 ];
