import sys
import os

# Add the src directory to Python path so we can import modules
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_dir)

import model.sampler
import plotting.image_plots
import numpy as np

# Hook into and use the in-built sampler
model_sampler = model.sampler.Sampler()

# Simply add functionality for discrete variation
image_sizes = [40+20*x for x in range(20)]  # note that image size must be multiple of 4

for size in image_sizes:
    # Create the samples using in-built quick_sample
    samples = model_sampler.quick_sample(
        "LOFAR_model",
        distribute_model=False,  # no access to multiple GPUs here; CPU functionality only
        n_samples=1,
        image_size=size,
        return_steps=False
    )

    # Plot only the final image the diffusion process seems pretty consistent w.r.t. parameters
    fig, ax = plotting.image_plots.plot_image_grid(samples[0])
    fig.savefig("varying_parameters/" + str(size) + ".png")