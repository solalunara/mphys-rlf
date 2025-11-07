import generate_fits_files; #generated prep
import dataset_h5_to_fits; #dataset prep
import utils.paths;
from pybdsf_analysis.image_analyzer import ImageAnalyzer;

#Again we want it so that by importing this file we prepare all neccesary data

generated_analyzer = ImageAnalyzer( utils.paths.GENERATED_SUBDIR, export_images=[ 'gaus_model', 'gaus_resid' ], catalog_format='fits' );
dataset_analyzer = ImageAnalyzer( utils.paths.DATASET_SUBDIR, export_images=[ 'gaus_model', 'gaus_resid' ], catalog_format='fits' );

generated_analyzer.AnalyzeAllFITSInInput();
dataset_analyzer.AnalyzeAllFITSInInput();