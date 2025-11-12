import scripts.pybdsf_run_analysis;
import utils.paths;
from pybdsf_analysis.image_analyzer import ImageAnalyzer;
import numpy as np;
from pathlib import PurePath;
import matplotlib.pyplot as plt;

if __name__ == '__main__':
    dataset_resid_analyzer = ImageAnalyzer( f"{utils.paths.DATASET_SUBDIR}/gaus_resid", fits_input_dir=utils.paths.EXPORT_IMAGE_PARENT );
    generated_resid_analyzer = ImageAnalyzer( f"{utils.paths.GENERATED_SUBDIR}/gaus_resid", fits_input_dir=utils.paths.EXPORT_IMAGE_PARENT );

    # Delta - summed clipped residuals, per image
    dataset_resid_values = dataset_resid_analyzer.GetPixelValues();
    generated_resid_values = generated_resid_analyzer.GetPixelValues();

    drv_clipped = np.where( dataset_resid_values > 0, dataset_resid_values, 0 );
    grv_clipped = np.where( generated_resid_values > 0, generated_resid_values, 0 );

    dataset_delta = np.sum( drv_clipped, axis=(1,2) );
    generated_delta = np.sum( grv_clipped, axis=(1,2) );

    # Scaled flux 
    dataset_analyzer = ImageAnalyzer( utils.paths.DATASET_SUBDIR );
    generated_analyzer = ImageAnalyzer( utils.paths.GENERATED_SUBDIR );

    dataset_scaled_flux = dataset_analyzer.GetScaledFlux();
    generated_scaled_flux = generated_analyzer.GetScaledFlux();

    # Combined points
    dataset_pts = np.array( (dataset_scaled_flux, dataset_delta) );
    generated_pts = np.array( (generated_scaled_flux, generated_delta) );

    plt.scatter( dataset_scaled_flux, dataset_delta, label='dataset', color='b' );
    plt.scatter( generated_scaled_flux, generated_delta, label='generated', color='g' );
    plt.legend();
    plt.title( 'Scaled flux vs summed residuals' );
    plt.show();
    plt.savefig( 'scatter.png' );
