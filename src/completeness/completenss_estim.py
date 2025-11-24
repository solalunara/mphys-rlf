from astropy.io import fits;
import numpy as np;
import pybdsf_analysis.pybdsf_run_analysis;
import scipy.stats;
from pybdsf_analysis.recursive_file_analyzer import RecursiveFileAnalyzer;
import utils.paths;
from utils.distributed import DistributedUtils;
import random;
from pybdsf_analysis.log_analyzer import LogAnalyzer;
import pybdsf_analysis.log_analyzer as la;
import pybdsf_analysis.recursive_file_analyzer as rfa;
from tqdm import tqdm;
import astropy.stats;
import pandas as pd;
import matplotlib.pyplot as plt;
from pathlib import Path;
from pybdsf_analysis.image_analyzer import ImageAnalyzer;
import h5py;
from files.dataset import LOFAR_DATA_PATH;
from sklearn.preprocessing import PowerTransformer;
from completeness.img_data_arrays import ImageDataArrays;

rms_LOFAR = 71e-6 * 1e3;
beam_width_LOFAR = ImageAnalyzer.LOFAR_process_arg_defaults[ 'process_beam' ][ :-1 ];
beam_area_LOFAR = beam_width_LOFAR[ 0 ] * beam_width_LOFAR[ 1 ];

def get_noise(data):
    """
    from Cyril Tasse/kMS, courtesy of Wara
    """
    maskSup = 1e-7
    m = data[np.abs(data) > maskSup]
    rmsold = np.std(m)
    diff = 1e-1
    cut = 3.
    med = np.median(m)
    for _ in range(10):
        ind = np.where(np.abs(m - med) < rmsold * cut)[0]
        rms = np.std(m[ind])
        if np.abs((rms - rmsold)//rmsold) < diff: break
        rmsold = rms
    return rms

def masking(fits_data, threshold_level = 5.0):
    mean, median, std_dev = astropy.stats.sigma_clipped_stats(fits_data, sigma=3.0)
    
    # Calculate the threshold
    threshold = threshold_level * std_dev
    
    # Create a mask for values less than the threshold
    mask = fits_data < threshold
    
    # Set values less than the threshold to zero
    fits_data_nr = np.where(mask, 0, fits_data)

    return fits_data_nr

def create_noise_LOFAR(shape=(80,80), rms=rms_LOFAR):
    """
    Create a 2D patch of Gaussian noise with given RMS.
    """
    return np.random.normal(loc=0.0, scale=rms, size=shape)

def get_completeness_estim():
    plt.figure(figsize = (8, 5))
    for subdir in [ utils.paths.GENERATED_SUBDIR, utils.paths.DATASET_SUBDIR ]:
        images, resid_images, model_fluxes, peak_fluxes, sigma_clipped_means, sigma_clipped_rmsds = ImageDataArrays( subdir ).get_all_arrays();

        if subdir == utils.paths.DATASET_SUBDIR:
            NUM_MOCKS = 100;
            NUM_SCALE_FACTORS = 100;
            mock_fluxes = np.empty( (NUM_MOCKS*NUM_SCALE_FACTORS), dtype=float );
            detectable = np.empty( (NUM_MOCKS*NUM_SCALE_FACTORS), dtype=bool );
        else:
            mock_fluxes = np.empty( (images.shape[ 0 ]), dtype=float );
            detectable = np.empty( (images.shape[ 0 ]), dtype=bool );

        for i in tqdm( range( images.shape[ 0 ] if subdir == utils.paths.GENERATED_SUBDIR else NUM_MOCKS ), desc='Calculating mock images' ):
            #rms = image_rmss_actual[ random_image ];
            #noise_patch = resid_images[ random_image ];

            # Using rms=image_rmss_actual[ random_image ] is technically correct yet utterly useless
            # because the majority of the noise is from the artificial 1% noise added for pybdsf
            # TODO: Use raw LOFAR data so we can get rms locally based on strength of source, potential code commented above
            rms = rms_LOFAR;
            if subdir == utils.paths.DATASET_SUBDIR:
                # Mix a random data image with a random residual image
                flux_scale_factors = scipy.stats.loguniform.rvs( 10**(-3), 10**(2), size=NUM_SCALE_FACTORS );
                random_image = int( random.random() * images.shape[ 0 ] )
                for j in range( len( flux_scale_factors ) ):
                    flux_scale_factor = flux_scale_factors[ j ];
                    s_mock = flux_scale_factor * model_fluxes[ random_image ];
                    mock_fluxes[ i * NUM_MOCKS + j ] = s_mock;
                    mock_data = flux_scale_factor * images[ random_image ];
                    noise_patch = create_noise_LOFAR( rms=rms );
                    sim_data = noise_patch + mock_data;

                    peak_flux = np.max( sim_data );
                    threshold = 5 * rms;
                    detectable[ i * NUM_MOCKS + j ] = peak_flux >= threshold;

            else:
                mock_fluxes[ i ] = model_fluxes[ i ];
                noise_patch = create_noise_LOFAR( rms=rms );
                sim_data = noise_patch + images[ i ];

                peak_flux = np.max( sim_data );
                threshold = 5 * rms;
                detectable[ i ] = peak_flux >= threshold;


        test_mock = pd.DataFrame()
        #test_mock['mock_flux'] = mock_fluxes.ravel()
        #test_mock['detectable'] = detectable.ravel()
        test_mock['mock_flux'] = mock_fluxes;
        test_mock['detectable'] = detectable;

        # Define flux bins
        flux_bins = np.linspace( 0, 10, num=25 )
        bin_centers = 0.5 * (flux_bins[1:] + flux_bins[:-1])

        # Bin and count
        completeness = []   # to store completeness per bin
        total_counts = []   # optional: for diagnostics

        for i in range(len(flux_bins) - 1):
            # Select sources in this flux bin
            in_bin = (mock_fluxes >= flux_bins[i]) & (mock_fluxes < flux_bins[i + 1])

            n_detect = test_mock[ (test_mock['mock_flux'] >= flux_bins[i]) & (test_mock['mock_flux'] < flux_bins[i + 1]) ]
            
            if np.sum(in_bin) > 0:
                frac_recovered = np.sum(n_detect['detectable']) / np.sum(in_bin)
            else:
                frac_recovered = 0  

            completeness.append(frac_recovered)
            total_counts.append(np.sum(in_bin))
        conf_interval = astropy.stats.poisson_conf_interval( np.array( completeness ) * np.array( total_counts ), sigma=1.0, interval='frequentist-confidence' ) / np.array( total_counts );
        yerr = conf_interval[ 1 ] - conf_interval[ 0 ];
        # Plot completeness curve

        plt.errorbar( bin_centers, completeness, yerr, fmt='.', color='b' if subdir is utils.paths.DATASET_SUBDIR else 'g' );

        plt.plot(bin_centers, completeness, marker='.', label = f'{subdir} completeness', color='b' if subdir is utils.paths.DATASET_SUBDIR else 'g' )

        plt.xscale('log')
        plt.ylim(0, 1.1)
        plt.xlabel("Flux Density (mJy)")
        plt.ylabel("Completeness")
        plt.title("Flux Density Completeness Curve")
        plt.grid(True)
        plt.legend()
    plt.show()


if __name__ == "__main__":
    pybdsf_analysis.pybdsf_run_analysis.analyze_everything();

    du = DistributedUtils();
    du.last_task_only( 'get_completeness_estim', get_completeness_estim );

