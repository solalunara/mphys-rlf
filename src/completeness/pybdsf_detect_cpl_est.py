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
    N_NOISE_PATCHES = 5;
    for subdir in [ utils.paths.GENERATED_SUBDIR ]:
        images, resid_images, model_images, model_fluxes, peak_fluxes, sigma_clipped_means, sigma_clipped_rmsds = ImageDataArrays( subdir ).get_all_arrays();

        detectable = np.empty( (images.shape[ 0 ]), dtype=bool );

        # Define flux bins and get the average samples per bin for > 10 mJy (before we start having issues)
        flux_bins = np.logspace( -2, 2, num=25 )
        bin_centers = 0.5 * (flux_bins[1:] + flux_bins[:-1])
        total_counts = np.empty( len( flux_bins ) - 1, dtype=float )
        for i in range(len(flux_bins) - 1):
            in_bin = (model_fluxes >= flux_bins[i]) & (model_fluxes < flux_bins[i + 1])
            total_counts[ i ] = np.sum( in_bin )
        samples_per_bin_average = np.average( total_counts[ bin_centers > 10 ] )
        print( f'Average samples per bin >10mJy: {samples_per_bin_average}' )

        detectable = model_fluxes > 0
        detectable_model_fluxes = model_fluxes[ detectable ]

        # Bin and count
        completeness = np.empty( len( flux_bins ) - 1, dtype=float )

        for i in range(len(flux_bins) - 1):
            # Select sources in this flux bin
            detected_in_bin, = np.where( np.logical_and( detectable_model_fluxes >= flux_bins[i], detectable_model_fluxes < flux_bins[i + 1] ) )
            n_detect = detected_in_bin.shape[ 0 ]
            completeness[ i ] = n_detect / samples_per_bin_average

        # Handle confidence interval with poisson_conf_interval for total_counts = 0
        conf_interval = astropy.stats.poisson_conf_interval( completeness * samples_per_bin_average, interval='frequentist-confidence', sigma=1.0 );
        conf_interval[ :, total_counts != 0 ] /= total_counts[ total_counts != 0 ];
        yerr = conf_interval[ 1 ] - conf_interval[ 0 ];

        # Plot completeness curve

        plt.errorbar( bin_centers, completeness, yerr, fmt='.', color='b' if subdir is utils.paths.DATASET_SUBDIR else 'g' );

        plt.plot(bin_centers, completeness, marker='.', label = f'{subdir} completeness', color='b' if subdir is utils.paths.DATASET_SUBDIR else 'g' )

        plt.xscale('log')
        plt.ylim(0, 1.1)
        plt.xlabel("Flux Density (mJy)")
        plt.ylabel("Samples per Bin")
        plt.title("Normalized PyBDSF Resolved Sources per Integrated Flux Bin")
        plt.grid(True)
        plt.legend()
    plt.show()
    plt.savefig( 'cplestim.png' );


if __name__ == "__main__":
    pybdsf_analysis.pybdsf_run_analysis.analyze_everything();

    du = DistributedUtils();
    du.last_task_only( 'get_completeness_estim', get_completeness_estim );

