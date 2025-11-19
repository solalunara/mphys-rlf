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

rms_LOFAR = 71e-6;
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

    for subdir in [ utils.paths.DATASET_SUBDIR, utils.paths.GENERATED_SUBDIR ]:
        data_files = RecursiveFileAnalyzer( utils.paths.FITS_PARENT / subdir );
        log_analyzer = LogAnalyzer( subdir );
        model_fluxes, log_analyzer_inds = log_analyzer.for_each( la.get_model_flux, return_nums=True );
        model_fluxes = np.array( model_fluxes ) * 1000; #Units of mJy
        sigma_clipped_means = np.array( log_analyzer.for_each( la.get_sigma_clipped_mean ) );
        sigma_clipped_rms = np.array( log_analyzer.for_each( la.get_sigma_clipped_rms ) );
        images, data_inds = data_files.for_each( rfa.get_fits_primaryhdu_data, pattern=r'.*?image(\d+)\.fits$', return_nums=True );
        delt1 = np.array( data_files.for_each( rfa.get_fits_primaryhdu_header, pattern=r'.*?image(\d+)\.fits$', kwargs=dict( key='CDELT1' ) ) );
        delt2 = np.array( data_files.for_each( rfa.get_fits_primaryhdu_header, pattern=r'.*?image(\d+)\.fits$', kwargs=dict( key='CDELT2' ) ) );
        images = np.array( images );

        
        # Make it so fmr and images match indices
        intersect, comm1, comm2 = np.intersect1d( log_analyzer_inds, data_inds, return_indices=True );
        model_fluxes = model_fluxes[ comm1 ];
        sigma_clipped_means = sigma_clipped_means[ comm1 ];
        sigma_clipped_rms = sigma_clipped_rms[ comm1 ];
        images = images[ comm2 ];
        delt1 = delt1[ comm2 ];
        delt2 = delt2[ comm2 ];

        residual_files = RecursiveFileAnalyzer( utils.paths.PYBDSF_EXPORT_IMAGE_PARENT / subdir / 'gaus_resid' );
        residual_images, residual_indexes = residual_files.for_each( rfa.get_fits_primaryhdu_data, pattern=r'.*?image(\d+)\.fits$', return_nums=True )
        residual_images = np.array( residual_images );
        
        # Make it so we can assume residual_images[ i ] is the residual of images[ i ]
        residual_images = residual_images[ np.isin( residual_indexes, intersect, assume_unique=True ) ];

        NUM_MOCKS = 50;
        NUM_SCALE_FACTORS = 100;
        mock_fluxes = np.empty( (NUM_MOCKS, NUM_SCALE_FACTORS), dtype=float );
        detectable = np.empty( (NUM_MOCKS, NUM_SCALE_FACTORS), dtype=bool );
        for i in tqdm( range( NUM_MOCKS ), desc='Calculating mock images' ):
            flux_scale_factors = scipy.stats.loguniform.rvs( 10**(-3), 10**(1), size=NUM_SCALE_FACTORS );
            # Mix a random data image with a random residual image
            random_image = int( random.random() * images.shape[ 0 ] );
            model_flux = model_fluxes[ random_image ];
            rms = sigma_clipped_rms[ random_image ];

            this_image_delt1 = delt1[ random_image ];
            this_image_delt2 = delt2[ random_image ];
            area_per_pix = this_image_delt1 * this_image_delt2;

            # Jy/beam = 1000 mJy/pix beam/pix = 1000 beam/area area/pix mJy/pix
            data_image = images[ random_image ] * 1000 * beam_area_LOFAR / area_per_pix; # Jy/beam -> mJy/pix

            for j in range( len( flux_scale_factors ) ):
                flux_scale_factor = flux_scale_factors[ j ];
                s_mock = flux_scale_factor * model_flux;
                mock_fluxes[ i, j ] = s_mock;
                mock_data = flux_scale_factor * data_image;

                rms_ori = get_noise( data_image );
                rms_mock = get_noise( mock_data );

                random_residual = int( random.random() * residual_images.shape[ 0 ] );
                noise_patch = create_noise_LOFAR( rms=rms );
                #noise_patch = residual_images[ random_image ]

                sim_data = noise_patch + mock_data;
                clean_data = masking( sim_data, threshold_level = 3 );

                peak_flux = np.max( sim_data );
                threshold = 5 * rms;
                detectable[ i, j ] = peak_flux >= threshold;

        test_mock = pd.DataFrame()
        test_mock['mock_flux'] = mock_fluxes.ravel()
        test_mock['detectable'] = detectable.ravel()

        # Define flux bins
        flux_bins = np.logspace(-2, 5, num=25)  
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

