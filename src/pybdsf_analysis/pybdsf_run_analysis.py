import pybdsf_analysis.generate_fits_files
import pybdsf_analysis.dataset_h5_to_fits
import utils.paths
from pybdsf_analysis.image_analyzer import ImageAnalyzer
import utils.parameters
from utils.logging import get_logger

logger = get_logger( __name__ )

def analyze_dataset():
    logger.info( 'Preparing dataset' )
    pybdsf_analysis.dataset_h5_to_fits.single_node_validate_LOFAR_fits_images( utils.parameters.CLEAN_LOFAR_FITS_IMAGES, utils.parameters.LOFAR_FITS_COUNT_CUTOFF, utils.parameters.BINS_ARRAY )
    logger.info( 'Analyzing dataset' )
    dataset_analyzer = ImageAnalyzer( utils.paths.DATASET_SUBDIR, export_images=[ 'gaus_model', 'gaus_resid' ], catalog_format='fits' )
    dataset_analyzer.analyze_all_FITS_in_input()

def analyze_sampled():
    logger.info( 'Preparing samples' )
    pybdsf_analysis.generate_fits_files.sample( utils.parameters.FITS_SAMPLING_ARGS )
    logger.info( 'Analyzing samples' )
    generated_analyzer = ImageAnalyzer( utils.paths.GENERATED_SUBDIR, export_images=[ 'gaus_model', 'gaus_resid' ], catalog_format='fits' )
    generated_analyzer.analyze_all_FITS_in_input()

def analyze_everything():
    analyze_sampled()
    analyze_dataset()

if __name__ == '__main__':
    analyze_everything()
