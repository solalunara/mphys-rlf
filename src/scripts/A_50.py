import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add the src directory to Python path so we can import modules
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_dir)

import model.sampler
import plotting.image_plots


class A_50:
    def __init__(self):
        pass

    def create_A_50_list(self, image):
        # Create a structured array to hold pixel positions and brightness
        A_50 = np.array([])

        # Add positional information to every pixel
        positions = np.array(np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))).T.reshape(-1, 2)
        brightness = image.flatten()
        pixels = np.hstack((positions, brightness[:, None]))

        # Sort pixels by brightness in descending order
        sorted_pixels = pixels[np.argsort(-pixels[:, 2])]

        # Calculate total brightness
        total_brightness = np.sum(image)

        # Add pixels to A_50 until total brightness exceeds 50%
        cumulative_brightness = 0
        for pixel in sorted_pixels:
            A_50 = np.append(A_50, pixel)
            cumulative_brightness += pixel[2]
            if cumulative_brightness > 0.5 * total_brightness:
                break

        return A_50

    def plot_a50_contour(self, image, A50):
        # Create a50 contour level
        a50_level = [np.min(A_50)]

        x_mesh, y_mesh = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        contour = plt.contour(x_mesh, y_mesh, image, levels=a50_level, colors=['white'])
        # Create meshgrid for contour plotting
        x_mesh, y_mesh = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        a50_plot = plt.contour(x_mesh, y_mesh, image, levels=a50_level, colors=['white'])

        fig.savefig("a50contour.png")
        print()
        return contour

# Run the sampler and produce an image
model_sampler = model.sampler.Sampler()
samples = model_sampler.quick_sample(
    "LOFAR_model",
    distribute_model=False,
    n_samples=1,
    image_size=80,
    return_steps=False
)
fig, ax = plotting.image_plots.plot_image_grid( samples[ 0 ] )

# Run the A50 sorting algorithm
# Create a structured array to hold pixel positions and brightness
A_50 = np.array([])

# Grab the image array from the samples
image = samples[0][0]