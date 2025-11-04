import sys
import os
import bdsf;
from astropy.io import fits;
from astropy.io.fits import ImageHDU;
import numpy as np;
import matplotlib.pyplot as plt;
import h5py;
import math;
import argparse;

# Add the src directory to Python path so we can import modules
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_dir)


import model.sampler;
import plotting.image_plots;
from scripts.image_analyzer import ImageAnalyzer;


if __name__ == "__main__":
    parser = argparse.ArgumentParser();
    parser.add_argument( "-b", "--batch-size", help="The number of batches to be sampled at a time - default 100", type=int, default=100 );
    parser.add_argument( "-n", "--n-samples", help="The number of samples to generate - default 10000", type=int, default=60000 );
    parser.add_argument( "-t", "--timesteps", help="The number of timesteps in sampling - default 25", type=int, default=25 );
    parser.add_argument( "-c", "--use-cpu", help="Whether or not to use CPU and RAM for sampling, as opposed to using avaliable GPUs", action=argparse.BooleanOptionalAction );
    parser.add_argument( "-u", "--leave-unscaled", help="Whether or not to leave the images unscaled instead of scaling them 0-1", action=argparse.BooleanOptionalAction );
    parser.add_argument( "-sz", "--bin-size", help="How large the bins the generated images are sorted into are - default 10000", type=int, default=10000 );
    parser.add_argument( "-i", "--initial_count", help="What value to start generation at (affects file saving) - default 0", type=int, default=0 );
    args = parser.parse_args();

    batch_size = args.batch_size;
    n_samples = args.n_samples;
    timesteps = args.timesteps;
    distribute_model = not args.use_cpu;
    scale_results = not args.leave_unscaled;
    bin_size = args.bin_size;
    initial_count = args.initial_count;

    model_sampler = model.sampler.Sampler();
    samplecount = initial_count;
    image_analyzer = ImageAnalyzer( "generated" );
    while samplecount < n_samples:
        samples = model_sampler.quick_sample(
            "LOFAR_model",
            distribute_model=distribute_model,
            n_samples=batch_size,
            timesteps=timesteps
        );

        for i in range( samples.shape[ 0 ] ):
            image = samples[ i, -1, 0, :, : ];

            # the images in the dataset *are* selected by the process in the paper but *are not* scaled 0-1
            # here we do that scaling, if we so choose
            if scale_results:
                im_max = np.max( image );
                im_min = np.min( image );
                if im_min < 0:
                    image = np.where( image > 0, image, 0 );
                image = ( image - im_min ) / ( im_max - im_min );

            lower_bound = int( math.floor( ( samplecount + i ) / bin_size ) * bin_size );
            upper_bound = int( math.ceil( ( samplecount + i + 1 ) / bin_size ) * bin_size ) - 1;
            postfix = f"{lower_bound}-{upper_bound}/image{samplecount+i}.fits";
            image_analyzer.SaveImageToFITS( image, postfix );

        samplecount += batch_size;
