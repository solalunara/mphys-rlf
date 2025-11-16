from astropy.io import fits;
import numpy as np;
import pybdsf_analysis.pybdsf_run_analysis;
import scipy.stats;
from pybdsf_analysis.recursive_file_analyzer import RecursiveFileAnalyzer;
import utils.paths;
from utils.distributed import DistributedUtils;
import random;
from pybdsf_analysis.log_analyzer import LogAnalyzer;
from tqdm import tqdm;
import astropy.stats;
import pandas as pd;
import matplotlib.pyplot as plt;


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

def get_completeness_estim():

    for subdir in [ utils.paths.DATASET_SUBDIR, utils.paths.GENERATED_SUBDIR ]:
        data_files = RecursiveFileAnalyzer( utils.paths.FITS_PARENT / subdir );
        log_analyzer = LogAnalyzer( subdir );
        fmr, fmr_inds = log_analyzer.fmr_data( True );
        fmr = np.array( fmr );
        data_unwrapped_files = data_files.get_unwrapped_list( pattern=r'.*?image(\d+)\.fits$', return_nums=True );

        images = np.empty( (len( data_unwrapped_files ), 80, 80) );
        img_inds = np.empty( (len( data_unwrapped_files )) );
        i = 0;
        for file in tqdm( data_unwrapped_files, desc='Gathering file data...' ):
            with fits.open( str( file[ 0 ] ) ) as hdul:
                images[ i ] = hdul[ 0 ].data;
            img_inds[ i ] = file[ 1 ];
            i += 1;
        
        # Make it so fmr and images match indices
        intersect, comm1, comm2 = np.intersect1d( fmr_inds, img_inds, return_indices=True );
        fmr = fmr[ comm1 ];
        images = images[ comm2 ];

        residual_files = RecursiveFileAnalyzer( utils.paths.PYBDSF_EXPORT_IMAGE_PARENT / subdir / 'gaus_resid' );
        residual_image_files = residual_files.get_unwrapped_list( pattern=r'.*?\.fits$' );
        residual_images = np.empty( (len( residual_image_files ), 80, 80) );
        i = 0;
        for file in tqdm( residual_image_files, desc='Gathering file data...' ):
            with fits.open( str( file ) ) as hdul:
                residual_images[ i ] = hdul[ 0 ].data;
            i += 1;

        NUM_MOCKS = 100;
        NUM_SCALE_FACTORS = 100;
        mock_fluxes = [ None ] * NUM_MOCKS * NUM_SCALE_FACTORS;
        detectable = [ None ] * NUM_MOCKS * NUM_SCALE_FACTORS;
        for i in range( NUM_MOCKS ):
            flux_scale_factors = scipy.stats.loguniform.rvs( 10**(-5), 10**3, size=NUM_SCALE_FACTORS );
            # Mix a random data image with a random residual image
            random_image = int( random.random() * images.shape[ 0 ] );
            data_image = images[ random_image ];
            fmr_image = fmr[ random_image ];

            for j in range( len( flux_scale_factors ) ):
                flux_scale_factor = flux_scale_factors[ j ];
                s_mock = flux_scale_factor * fmr_image[ 0 ];
                mock_fluxes[ i * NUM_MOCKS + j ] = s_mock;
                mock_data = flux_scale_factor * data_image;

                rms_ori = get_noise( data_image );
                rms_mock = get_noise( mock_data );

                random_residual = int( random.random() * residual_images.shape[ 0 ] );
                noise_patch = residual_images[ random_residual ];

                sim_data = noise_patch + mock_data;
                clean_data = masking( sim_data, threshold_level = 3 );

                peak_flux = np.max( sim_data );
                threshold = 5 * rms_ori;
                detectable[ i * NUM_MOCKS + j ] = peak_flux >= threshold;

        test_mock = pd.DataFrame()
        test_mock['mock_flux'] = mock_fluxes
        test_mock['detectable'] = detectable

        # Define flux bins
        flux_bins = np.logspace(-5, 3, num=25)  
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
        # Plot completeness curve

        plt.figure(figsize = (8, 5))

        plt.plot(bin_centers, completeness, marker='.', label = 'Completeness')

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

