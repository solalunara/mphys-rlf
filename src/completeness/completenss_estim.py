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
    N_NOISE_PATCHES = 100;
    for subdir in [ utils.paths.GENERATED_SUBDIR ]:
        images, resid_images, model_images, model_fluxes, peak_fluxes, sigma_clipped_means, sigma_clipped_rmsds = ImageDataArrays( subdir ).get_all_arrays();

        mock_fluxes = np.empty( (images.shape[ 0 ]*N_NOISE_PATCHES), dtype=float );
        detectable = np.empty( (images.shape[ 0 ]*N_NOISE_PATCHES), dtype=bool );


        for i in tqdm( range( images.shape[ 0 ] ), desc='Calculating mock images' ):
            #rms = image_rmss_actual[ random_image ];
            #noise_patch = resid_images[ random_image ];

            # Using rms=image_rmss_actual[ random_image ] is technically correct yet utterly useless
            # because the majority of the noise is from the artificial 1% noise added for pybdsf
            # TODO: Use raw LOFAR data so we can get rms locally based on strength of source, potential code commented above
            rms = rms_LOFAR;
            mock_fluxes[ i:(i+N_NOISE_PATCHES) ] = model_fluxes[ i ][ np.newaxis ];
            noise_patches = create_noise_LOFAR( shape=(N_NOISE_PATCHES,80,80), rms=rms );
            sim_data = noise_patches + images[ i ][ np.newaxis, :, : ];

            peak_fluxes = np.max( sim_data, axis=(1,2) );
            threshold = 5 * rms;
            detectable[ i:(i+N_NOISE_PATCHES) ] = peak_fluxes >= threshold;


        test_mock = pd.DataFrame()
        #test_mock['mock_flux'] = mock_fluxes.ravel()
        #test_mock['detectable'] = detectable.ravel()
        test_mock['mock_flux'] = mock_fluxes;
        test_mock['detectable'] = detectable;

        # Define flux bins
        flux_bins = np.logspace( -2, 2, num=25 )
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

        # Handle confidence interval with poisson_conf_interval for total_counts = 0
        total_counts = np.array( total_counts );
        nonzero_counts = total_counts > 0;
        total_counts = np.where( nonzero_counts, total_counts, 1e-10 );
        conf_interval = astropy.stats.poisson_conf_interval( np.array( completeness ) * total_counts, sigma=1.0, interval='frequentist-confidence' ) / total_counts;
        conf_interval[ :, nonzero_counts ] = 0;
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
    plt.savefig( 'cplestim.png' );


if __name__ == "__main__":
    pybdsf_analysis.pybdsf_run_analysis.analyze_everything();

    du = DistributedUtils();
    du.last_task_only( 'get_completeness_estim', get_completeness_estim );

