import sys
import os

# Add the src directory to Python path so we can import modules
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_dir)

import model.sampler
import plotting.image_plots

class SimpleSample:
    def __init__(self):
        # Hook into the in-built sampler
        self.model_sampler = model.sampler.Sampler()

    def run(self, file_name="sample.png", **kwargs):
        # Generate a sample according to some parameters
        samples = self.model_sampler.quick_sample(**kwargs)

        # Save to a file
        fig, ax = plotting.image_plots.plot_image_grid(samples[0])
        fig.savefig(file_name)

    def quick_run(self):
        # Generate a sample with default parameters
        self.run(file_name="sample.png", model_name="LOFAR_model", distribute_model=False, n_samples=1, image_size=80)

# if __name__ == "__main__":
#     simple_sample = SimpleSample()
#     simple_sample.quick_run()