import sys
import os

# Add the src directory to Python path so we can import modules
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_dir)


import model.sampler
import plotting.image_plots

model_sampler = model.sampler.Sampler();

samples = model_sampler.quick_sample(
    "LOFAR_model",
    distribute_model=False,
    n_samples=1,
    image_size=80,
);

fig, ax = plotting.image_plots.plot_image_grid( samples[ 0 ] )
fig.savefig( "sample.png" );