import scripts.pybdsf_run_analysis;
import utils.paths;
from pybdsf_analysis.image_analyzer import ImageAnalyzer;
import numpy as np;
from pathlib import PurePath;
import matplotlib.pyplot as plt;

if __name__ == '__main__':
    scripts.pybdsf_run_analysis.analyze_everything();

    for subdir in [ utils.paths.DATASET_SUBDIR, utils.paths.GENERATED_SUBDIR ]:
        resid_analyzer = ImageAnalyzer( f"{subdir}/gaus_resid", fits_input_dir=utils.paths.PYBDSF_EXPORT_IMAGE_PARENT, write_catalog=False );

        # Delta - summed clipped residuals, per image
        resid_values, resid_indexes = resid_analyzer.GetPixelValues( True );
        rv_clipped = np.where( resid_values > 0, resid_values, 0 );
        delta = np.sum( rv_clipped, axis=(1,2) );

        # Scaled flux 
        analyzer = ImageAnalyzer( subdir, write_catalog=False );
        scaled_flux, scaled_indexes = analyzer.GetScaledFlux( True );

        # Combined points
        intersect, comm1, comm2 = np.intersect1d( resid_indexes, scaled_indexes, return_indices=True );
        matches_scaled = np.isin( scaled_indexes, resid_indexes, assume_unique=True );
        pts = np.array( (delta[ comm1 ], scaled_flux[ comm2 ]) );

        plt.scatter( pts[ 1 ], pts[ 0 ], label=subdir, 
                     color='g' if subdir == utils.paths.GENERATED_SUBDIR else 'b' );

    plt.xlabel( 'Scaled Flux' );
    plt.ylabel( 'Image Delta' );
    plt.yscale( 'log' );
    plt.legend();
    plt.title( 'Scaled flux vs summed residuals' );
    plt.show();
    plt.savefig( 'scatter.png' );
