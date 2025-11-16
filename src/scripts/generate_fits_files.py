# This file was created by Ashley and Luna. It provides a complete application that can be used to sample
# images according to the LOFAR model with certian parameters, and it provides a function that can be used
# to access that application through other files. This application can be distributed across multiple nodes.

import numpy as np;
import math;
import argparse;
import model.sampler;
from pybdsf_analysis.image_analyzer import ImageAnalyzer, RecursiveFileAnalyzer;
import utils.paths;
import utils.logging;
import torch;
import os;
import files.sampling;
from utils.parameters import FITS_SAMPLING_ARGS;
import sys;
from files.dataset import LOFAR_DATA_PATH, single_node_download_dataset;
from utils.distributed import DistributedUtils;
import h5py;
import utils.paths as pth;
import logging;
from pathlib import Path;

def get_h5_maxvals( outfile, infile ):
    with h5py.File(infile, "r") as f:
        max_vals = np.max(f["images"][:], axis=(1, 2))
    np.save( outfile, max_vals );    

def get_path_from_index( index: int, bin_size: int ):
    lower_bound = int( math.floor( ( index ) / bin_size ) * bin_size );
    upper_bound = int( math.ceil( ( index + 1 ) / bin_size ) * bin_size ) - 1;
    postfix = [ f"{lower_bound}-{upper_bound}", f"image{index}.fits" ];
    full_image_path = ( utils.paths.FITS_PARENT / utils.paths.GENERATED_SUBDIR ).joinpath( *postfix );
    return full_image_path, postfix;

def sample( parameter_args ):
    files.sampling.single_node_prepare_for_sampling();

    logger = utils.logging.get_logger( __name__, logging.DEBUG );

    parser = argparse.ArgumentParser();
    parser.add_argument( "-b", "--batch-size", help="The number of batches to be sampled at a time - default 100", type=int, default=100 );
    parser.add_argument( "-n", "--n-samples", help="The number of samples to generate - default 10000", type=int, default=10000 );
    parser.add_argument( "-t", "--timesteps", help="The number of timesteps in sampling - default 25", type=int, default=25 );
    parser.add_argument( "-c", "--use-cpu", help="Whether or not to use CPU and RAM for sampling, as opposed to using avaliable GPUs", action='store_true' );
    parser.add_argument( "-p", "--preserve-values", help="Whether or not to preserve unscaled image values. By default images are scaled 0-1", action='store_true' );
    parser.add_argument( "-sz", "--bin-size", help="How large the bins the generated images are sorted into are - default 10000", type=int, default=10000 );
    args = parser.parse_args( parameter_args ); #will automatically read from the command line if passed, else use defaults

    #Do a sampling loop of batch_size samples and save them to the disk as they're generated, until we reach n_samples
    model_sampler = model.sampler.Sampler( n_samples=args.batch_size, timesteps=args.timesteps, distribute_model=(not args.use_cpu) );


    #SLURM distribution w/ batching
    du = DistributedUtils();
    task_count = du.get_task_count();
    task_id = du.get_task_id();
    n_samples = args.n_samples;
    bin_start = int( task_id / task_count * n_samples );
    bin_end = int( ( task_id + 1 ) / task_count * n_samples );
    if task_id + 1 == task_count:
        bin_end = n_samples; #just in case the float->int conversion is messy
    logger.debug( 'bin_end=%i, bin_start=%i, n_samples=%i', bin_end, bin_start, n_samples );

    # Figure out initial count based on number of fits files already in the directory
    logger.debug( 'Getting initial count...' );
    initial_count = 0;
    generated_images_dir = utils.paths.FITS_PARENT / utils.paths.GENERATED_SUBDIR;
    if generated_images_dir.exists():
        analyzer = RecursiveFileAnalyzer( generated_images_dir );
        initial_count = len( analyzer.GetUnwrappedList( None, r'.*?image(\d+)\.fits$', (bin_start, bin_end) ) );
    n_samples_in_bin = bin_end - bin_start;
    logger.debug( 'Got initial count %i, requested samples in this bin %i', initial_count, n_samples_in_bin );

    n_samples_to_generate = n_samples_in_bin - initial_count;
    if n_samples_to_generate <= 0:
        logger.info( 'Skipping bin %i-%i, nothing to do', bin_start, bin_end );
        return;


    # Get a distribution of scaled max fluxes from the lofar data
    # This requires:
    #   1/ The dataset is downloaded
    #   2/ The max values of the dataset to be saved to a file using only one node while the rest wait,
    #   3/ The max values file to be copied so it can be simultaneously read by multiple nodes
    #   4/ Those max values to be passed to get_fpeak_model_dist
    logger.debug( 'Making sure we have the dataset...' );
    single_node_download_dataset();
    logger.debug( 'Dataset downloaded. Saving dataset maxvals to %s...', pth.MAXVALS_PARENT/'maxvals.npy' );
    du.single_task_only_forcewait( 'get_h5_maxvals', get_h5_maxvals, 0, pth.MAXVALS_PARENT/'maxvals.npy', str( LOFAR_DATA_PATH ) );
    logger.debug( 'Array saved. Copying array for distributed data processing to %i files...', task_count );
    du.copy_file_for_multiple_nodes( pth.MAXVALS_PARENT / 'maxvals.npy' );
    logger.debug( 'Files copied. Loading file %i', task_id );
    data = np.load( pth.MAXVALS_PARENT / f'maxvals_{task_id}.npy' );
    fpeak_model_dist = model_sampler.get_fpeak_model_dist( None, max_vals=data );
    logger.debug( 'Done with shared file IO' );

    sample_generated_count = 0;
    sample_index = bin_start;
    image_analyzer = ImageAnalyzer( utils.paths.GENERATED_SUBDIR );
    while sample_generated_count < n_samples_to_generate:
        batch_size = min( args.batch_size, n_samples_to_generate - sample_generated_count ); #to not double-generate at the borders
        fpeak_model_values = torch.from_numpy( fpeak_model_dist( batch_size )[ :, np.newaxis ] );
        samples = model_sampler.quick_sample( utils.paths.LOFAR_MODEL_NAME, labels=fpeak_model_values, n_samples=batch_size, distribute_model=(not args.use_cpu) );
        sample_generated_count += batch_size;

        for i in range( samples.shape[ 0 ] ):
            image = samples[ i, -1, 0, :, : ];

            # the images in the dataset *are* selected by the process in the paper but *are not* scaled 0-1
            # here we do that scaling, if we so choose
            if not args.preserve_values:
                im_max = np.max( image );
                im_min = np.min( image );
                if im_min < 0:
                    image = np.where( image > 0, image, 0 );
                image = ( image - im_min ) / ( im_max - im_min );

            fscaled = fpeak_model_values.numpy()[ i, 0 ];

            full_image_path, postfix = get_path_from_index( sample_index, args.bin_size );
            while full_image_path.exists():
                sample_index += 1;
                full_image_path, postfix = get_path_from_index( sample_index, args.bin_size );
            image_analyzer.SaveImageToFITS( image, postfix, fscaled );

            if sample_index > bin_end:
                logger.error( 'Sample index %i has gone outside allowed value %i', sample_index, bin_end );
            elif sample_index == bin_end:
                logger.info( 'Sample index %i has reached bin end %i - generated sample count %i/%i', sample_index, bin_end, sample_generated_count, n_samples_to_generate );

if __name__ == '__main__':
    sample( sys.argv[ 1: ] );