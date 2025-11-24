from pybdsf_analysis.recursive_file_analyzer import RecursiveFileAnalyzer;
from pybdsf_analysis.log_analyzer import LogAnalyzer;
import numpy as np;
import pybdsf_analysis.recursive_file_analyzer as rfa;
import pybdsf_analysis.log_analyzer as la;
import utils.paths as pth;
from utils.distributed import DistributedUtils;
from pybdsf_analysis.power_transform import PeakFluxPowerTransformer;
from utils.logging import get_logger;


class ImageDataArrays:
    """
    A class to collect the image data arrays for images in subdir from the original files and from pybdsf analysis,
    which will be useful for calculating the completeness corrections. All units from the image data arrays are in mJy,
    though the input images are expected to be normalized 0-1 and the scaled peak flux values in Jy. All arrays reference
    the same image at the same index and have the same length in the first dimension, though the order is nonstandard.

    Parameters
    ----------
    subdir : str
        The subdirectory to generate the image data arrays for. It is assumed when running the program that the data, pybdsf
        log files, and pybdsf gaus_resid images are all prepared for the data to generate the arrays of, though if any
        of the three are not present for an image it will not be included in the arrays with no error
    """
    def __init__( self, subdir: str ):
        self.logger = get_logger( __name__ );
        self.du = DistributedUtils();

        # Log analyzer arrays
        log_analyzer = LogAnalyzer( subdir );
        normalized_model_fluxes, log_analyzer_inds = log_analyzer.for_each( la.get_model_flux, return_nums=True );
        normalized_model_fluxes = np.array( normalized_model_fluxes );
        sigma_clipped_means = np.array( log_analyzer.for_each( la.get_sigma_clipped_mean ) );
        sigma_clipped_rmsds = np.array( log_analyzer.for_each( la.get_sigma_clipped_rms ) );
        unclipped_rmsds = np.array( log_analyzer.for_each( la.get_rms ) );
        self.logger.debug( 'Log analyzer length: %i', len( log_analyzer_inds ) );

        # Data arrays
        data_files = RecursiveFileAnalyzer( pth.FITS_PARENT / subdir );
        images, data_inds = data_files.for_each( rfa.get_fits_primaryhdu_data, pattern=r'.*?image(\d+)\.fits$', return_nums=True );
        images = np.array( images );
        #delt1 = np.array( data_files.for_each( rfa.get_fits_primaryhdu_header, pattern=r'.*?image(\d+)\.fits$', kwargs=dict( key='CDELT1' ) ) );
        #delt2 = np.array( data_files.for_each( rfa.get_fits_primaryhdu_header, pattern=r'.*?image(\d+)\.fits$', kwargs=dict( key='CDELT2' ) ) );
        peak_fluxes_transformed = np.array( data_files.for_each( rfa.get_fits_primaryhdu_header, pattern=r'.*?image(\d+)\.fits$', kwargs=dict( key='FXSCLD' ) ) );
        self.logger.debug( 'Data files length: %i', len( data_inds ) );

        # Residual array
        residual_files = RecursiveFileAnalyzer( pth.PYBDSF_EXPORT_IMAGE_PARENT / subdir / 'gaus_resid' );
        residual_images, residual_indexes = residual_files.for_each( rfa.get_fits_primaryhdu_data, pattern=r'.*?image(\d+)\.fits$', return_nums=True )
        self.logger.debug( 'Gaussian residual files length: %i', len( residual_images ) );

        
        # Match indices of log_analyzer values and data_files values
        intersect, log_analyzer_inds, data_inds = np.intersect1d( log_analyzer_inds, data_inds, return_indices=True );
        normalized_model_fluxes = normalized_model_fluxes[ log_analyzer_inds ];
        sigma_clipped_means = sigma_clipped_means[ log_analyzer_inds ];
        sigma_clipped_rmsds = sigma_clipped_rmsds[ log_analyzer_inds ];
        unclipped_rmsds = unclipped_rmsds[ log_analyzer_inds ];
        images = images[ data_inds ];
        #delt1 = delt1[ data_files_inds ];
        #delt2 = delt2[ data_files_inds ];
        peak_fluxes_transformed = peak_fluxes_transformed[ data_inds ];

        # Also intersect the residual indexes with the other intersection
        intersect, prev_intersect_inds, residual_indexes = np.intersect1d( intersect, residual_indexes, return_indices=True );
        normalized_model_fluxes = normalized_model_fluxes[ prev_intersect_inds ];
        sigma_clipped_means = sigma_clipped_means[ prev_intersect_inds ];
        sigma_clipped_rmsds = sigma_clipped_rmsds[ prev_intersect_inds ];
        unclipped_rmsds = unclipped_rmsds[ prev_intersect_inds ];
        images = images[ prev_intersect_inds ];
        #delt1 = delt1[ prev_intersect_inds ];
        #delt2 = delt2[ prev_intersect_inds ];
        peak_fluxes_transformed = peak_fluxes_transformed[ prev_intersect_inds ];

        # Adjust raw (normalized) model fluxes from pybdsf so we have a small flux instead of 0
        normalized_model_fluxes = np.where( normalized_model_fluxes > 0, normalized_model_fluxes, 1e-10 );
    
        # Get the unscaled fluxes and unscale everything accordingly
        pt = PeakFluxPowerTransformer();
        peak_fluxes = pt.inverse_transform( peak_fluxes_transformed ) * 1000;
        image_scale_factors = peak_fluxes / np.max( images, axis=(1,2) ); #Scale from current image maxes (~1) to what the values should be as per peak fluxes
        unscaled_sigma_clipped_rmsds = sigma_clipped_rmsds * image_scale_factors;
        unscaled_sigma_clipped_means = sigma_clipped_rmsds * image_scale_factors;
        model_fluxes = normalized_model_fluxes * image_scale_factors;
        unscaled_images = images * image_scale_factors[ :, np.newaxis, np.newaxis ];
        unscaled_residual_images = np.array( residual_images ) * image_scale_factors[ :, np.newaxis, np.newaxis ];
        

        # Save unscaled variables to class
        self.images = unscaled_images;
        self.residual_images = unscaled_residual_images;
        self.model_fluxes = model_fluxes;
        self.peak_fluxes = peak_fluxes;
        self.sigma_clipped_means = unscaled_sigma_clipped_means;
        self.sigma_clipped_rmsds = unscaled_sigma_clipped_rmsds;

    def get_all_arrays( self ):
        """
        Function to return all arrays for lazy parsing. Order returned is:

        self.images, self.residual_images, self.model_fluxes, self.peak_fluxes, self.sigma_clipped_means, self.sigma_clipped_rmsds;
        """
        return self.images, self.residual_images, self.model_fluxes, self.peak_fluxes, self.sigma_clipped_means, self.sigma_clipped_rmsds;


