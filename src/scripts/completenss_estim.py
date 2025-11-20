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
    with h5py.File( LOFAR_DATA_PATH, "r") as f:
        max_vals = np.max(f["images"][:], axis=(1, 2))
    pt = PowerTransformer( method="box-cox" );
    pt.fit( max_vals.reshape(-1, 1) );

    plt.figure(figsize = (8, 5))

    for subdir in [ utils.paths.DATASET_SUBDIR, utils.paths.GENERATED_SUBDIR ]:
        data_files = RecursiveFileAnalyzer( utils.paths.FITS_PARENT / subdir );
        log_analyzer = LogAnalyzer( subdir );
        model_fluxes, log_analyzer_inds = log_analyzer.for_each( la.get_model_flux, return_nums=True );
        model_fluxes = np.array( model_fluxes ) * 1000; #Pybdsf log flux units are Jy convert to mJy
        sigma_clipped_means = np.array( log_analyzer.for_each( la.get_sigma_clipped_mean ) ); #already mJy
        sigma_clipped_rms = np.array( log_analyzer.for_each( la.get_sigma_clipped_rms ) ); #already mJy
        unclipped_rms = np.array( log_analyzer.for_each( la.get_rms ) ); #already mJy
        images, data_inds = data_files.for_each( rfa.get_fits_primaryhdu_data, pattern=r'.*?image(\d+)\.fits$', return_nums=True );
        delt1 = np.array( data_files.for_each( rfa.get_fits_primaryhdu_header, pattern=r'.*?image(\d+)\.fits$', kwargs=dict( key='CDELT1' ) ) );
        delt2 = np.array( data_files.for_each( rfa.get_fits_primaryhdu_header, pattern=r'.*?image(\d+)\.fits$', kwargs=dict( key='CDELT2' ) ) );
        fluxes_scaled = np.array( data_files.for_each( rfa.get_fits_primaryhdu_header, pattern=r'.*?image(\d+)\.fits$', kwargs=dict( key='FXSCLD' ) ) );
        images = np.array( images ) * 1000; # dataset/model images in Jy, convert to mJy

        
        # Make it so fmr and images match indices
        intersect, comm1, comm2 = np.intersect1d( log_analyzer_inds, data_inds, return_indices=True );
        model_fluxes = model_fluxes[ comm1 ];
        sigma_clipped_means = sigma_clipped_means[ comm1 ];
        sigma_clipped_rms = sigma_clipped_rms[ comm1 ];
        images = images[ comm2 ];
        delt1 = delt1[ comm2 ];
        delt2 = delt2[ comm2 ];
        fluxes_scaled = fluxes_scaled[ comm2 ];

        # Get the unscaled fluxes and rescale the images accordingly
        fluxes_unscaled = pt.inverse_transform( fluxes_scaled.reshape( -1, 1 ) )[ :, 0 ] * 1000;
        image_scale_factors = fluxes_unscaled / model_fluxes;
        image_rmss_actual = sigma_clipped_rms * image_scale_factors;
        scaled_images = images * image_scale_factors[ :, np.newaxis, np.newaxis ];

        # Get the residual images and scale them
        residual_files = RecursiveFileAnalyzer( utils.paths.PYBDSF_EXPORT_IMAGE_PARENT / subdir / 'gaus_resid' );
        residual_images, residual_indexes = residual_files.for_each( rfa.get_fits_primaryhdu_data, pattern=r'.*?image(\d+)\.fits$', return_nums=True )
        scaled_residual_images = np.array( residual_images ) * image_scale_factors[ :, np.newaxis, np.newaxis ];
        
        # Make it so we can assume scaled_residual_images[ i ] is the residual of images[ i ]
        scaled_residual_images = scaled_residual_images[ np.isin( residual_indexes, intersect, assume_unique=True ) ];

        """plt.scatter( fluxes_unscaled, image_rmss_actual );
        plt.xscale( 'log' );
        plt.yscale( 'log' );
        plt.xlabel( 'Flux (mJy)' );
        plt.ylabel( 'Background RMS (mJy)' );
        plt.show();"""

        NUM_MOCKS = 100;
        NUM_SCALE_FACTORS = 100;
        mock_fluxes = np.empty( (NUM_MOCKS, NUM_SCALE_FACTORS), dtype=float );
        detectable = np.empty( (NUM_MOCKS, NUM_SCALE_FACTORS), dtype=bool );
        for i in tqdm( range( NUM_MOCKS ), desc='Calculating mock images' ):
            flux_scale_factors = scipy.stats.loguniform.rvs( 10**(-3), 10**(0), size=NUM_SCALE_FACTORS );
            # Mix a random data image with a random residual image
            random_image = int( random.random() * images.shape[ 0 ] );

            # Using rms=image_rmss_actual[ random_image ] is technically correct yet utterly useless
            # because the majority of the noise is from the artificial 1% noise added for pybdsf
            #rms = image_rmss_actual[ random_image ];
            rms = rms_LOFAR;

            for j in range( len( flux_scale_factors ) ):
                flux_scale_factor = flux_scale_factors[ j ];
                s_mock = flux_scale_factor * fluxes_unscaled[ random_image ];
                mock_fluxes[ i, j ] = s_mock;
                mock_data = flux_scale_factor * scaled_images[ random_image ];

                #random_residual = int( random.random() * scaled_residual_images.shape[ 0 ] );
                #noise_patch = scaled_residual_images[ random_image ];
                noise_patch = create_noise_LOFAR( rms=rms );

                sim_data = noise_patch + mock_data;

                peak_flux = np.max( sim_data );
                threshold = 5 * rms;
                detectable[ i, j ] = peak_flux >= threshold;


        test_mock = pd.DataFrame()
        test_mock['mock_flux'] = mock_fluxes.ravel()
        test_mock['detectable'] = detectable.ravel()

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

