import numpy as np;
import math;
import argparse;
import model.sampler;
from pybdsf_analysis.image_analyzer import ImageAnalyzer, RecursiveFileAnalyzer;
import utils.paths;
from utils.paths import DEFAULT_GENERATION_ARGS;
import utils.logging;
import torch;
import os;

logger = utils.logging.get_logger( __name__ );

#We want to run the following script regardless of if we're the main or being imported
#thus importing this script ensures data preparation, similar to importing utils.paths

parser = argparse.ArgumentParser();
parser.add_argument( "-b", "--batch-size", help="The number of batches to be sampled at a time - default 100", type=int, default=DEFAULT_GENERATION_ARGS[ 'batch_size' ] );
parser.add_argument( "-n", "--n-samples", help="The number of samples to generate - default 10000", type=int, default=DEFAULT_GENERATION_ARGS[ 'n_samples' ] );
parser.add_argument( "-t", "--timesteps", help="The number of timesteps in sampling - default 25", type=int, default=DEFAULT_GENERATION_ARGS[ 'timesteps' ] );
parser.add_argument( "-c", "--use-cpu", help="Whether or not to use CPU and RAM for sampling, as opposed to using avaliable GPUs", action=argparse.BooleanOptionalAction );
parser.add_argument( "-p", "--preserve-values", help="Whether or not to preserve unscaled image values. By default images are scaled 0-1", action=argparse.BooleanOptionalAction );
parser.add_argument( "-sz", "--bin-size", help="How large the bins the generated images are sorted into are - default 10000", type=int, default=DEFAULT_GENERATION_ARGS[ 'bin_size' ] );
parser.add_argument( "-i", "--initial-count", help="What value to start generation at (affects file saving) - by default figure out dynamically from number of stored files", type=int, default=DEFAULT_GENERATION_ARGS[ 'initial_count' ] );
args = parser.parse_args(); #will automatically read from the command line if passed, else use defaults

#Figure out initial count based on number of fits files already in the directory, but let the user supercede this value
initial_count = 0;
generated_images_dir = utils.paths.FITS_PARENT / utils.paths.GENERATED_SUBDIR;
if generated_images_dir.exists() and args.initial_count == -1:
    analyzer = RecursiveFileAnalyzer( generated_images_dir );
    initial_count = len( analyzer.GetUnwrappedList( None, 'fits' ) );
if args.initial_count >= 0:
    initial_count = args.initial_count;

#Do a sampling loop of batch_size samples and save them to the disk as they're generated, until we reach n_samples
model_sampler = model.sampler.Sampler( n_samples=args.batch_size, timesteps=args.timesteps, distribute_model=(not args.use_cpu) );
fpeak_model_dist = None;
while fpeak_model_dist == None:
    try:
        fpeak_model_dist = model_sampler.get_fpeak_model_dist( str( utils.paths.LOFAR_DATA_PATH ) );
    except:
        logger.info( 'Could not lock file - probably in use by another array. Trying again...' );

#SLURM distribution w/ batching
task_count = int( os.environ.get( "SLURM_ARRAY_TASK_COUNT", 1 ) );
task_id = int( os.environ.get( "SLURM_ARRAY_TASK_ID", 0 ) );
n_samples = args.n_samples - initial_count;
bin_start = int( task_id / task_count * n_samples ) + initial_count;
bin_end = int( ( task_id + 1 ) / task_count * n_samples ) + initial_count;
if task_id + 1 == task_count:
    bin_end = n_samples + initial_count; #just in case the float->int conversion is messy

samplecount = bin_start;
image_analyzer = ImageAnalyzer( utils.paths.GENERATED_SUBDIR );
while samplecount < bin_end:
    batch_size = max( args.batch_size, bin_end - samplecount ); #to not double-generate at the borders
    fpeak_model_values = torch.from_numpy( fpeak_model_dist( batch_size )[ :, np.newaxis ] );
    samples = model_sampler.quick_sample( utils.paths.LOFAR_MODEL_NAME, labels=fpeak_model_values, n_samples=batch_size );

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

        fscaled = fpeak_model_values[ i ];

        lower_bound = int( math.floor( ( samplecount + i ) / args.bin_size ) * args.bin_size );
        upper_bound = int( math.ceil( ( samplecount + i + 1 ) / args.bin_size ) * args.bin_size ) - 1;
        postfix = f"{lower_bound}-{upper_bound}/image{samplecount+i}.fits";
        image_analyzer.SaveImageToFITS( image, postfix, fscaled );

    samplecount += args.batch_size;