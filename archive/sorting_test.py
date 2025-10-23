import time
import random
import numpy as np

# Start timing
start_time = time.time()

# Create a list of 10, 80x80 2d arrays with random brightness values
print("Creating data...")
data = [np.random.rand(80, 80) for _ in range(10)]
print("Data created in %.3f seconds." % (time.time() - start_time))


## LUNA'S METHOD
start_time = time.time()

for idx, array in enumerate(data):
    # Create a structured array to hold pixel positions and brightness
    A_50 = np.array([])

    # Execution time
    execution_time = time.time()

    # Add positional information to every pixel
    positions = np.array(np.meshgrid(np.arange(array.shape[0]), np.arange(array.shape[1]))).T.reshape(-1, 2)
    brightness = array.flatten()
    pixels = np.hstack((positions, brightness[:, None]))

    # Sort pixels by brightness in descending order
    sorted_pixels = pixels[np.argsort(-pixels[:, 2])]

    # Calculate total brightness
    total_brightness = np.sum(array)

    # Add pixels to A_50 until total brightness exceeds 50%
    cumulative_brightness = 0
    for pixel in sorted_pixels:
        A_50 = np.append(A_50, pixel)
        cumulative_brightness += pixel[2]
        if cumulative_brightness > 0.5 * total_brightness:
            print(f"Array {idx}: Total brightness in A_50 exceeds 50% of total brightness.")
            break

    print(f"Array {idx+1}: Processed in %.3f seconds." % (time.time() - execution_time))

    print(A_50)

print("All arrays processed in %.3f seconds." % (time.time() - start_time))


print()
print()


## ASHLEY'S METHOD
start_time = time.time()

# Iterate over each element in the list
for idx, array in enumerate(data):
    # Initialize an empty array A_50 to store pixel positions and brightness
    A_50 = np.array([])

    # Calculate the total brightness of the array
    total_brightness = np.sum(array)

    # Execution time
    execution_time = time.time()

    while True:
        # Find the position of the brightest pixel
        brightest_pos = np.unravel_index(np.argmax(array, axis=None), array.shape)

        # Add this pixel position and associated brightness to A_50
        brightness = array[brightest_pos]
        A_50 = np.append(A_50, [brightest_pos[0], brightest_pos[1], brightness])

        # Check if the total brightness in A_50 exceeds 50% of the total brightness of the array
        if np.sum(A_50[2::3]) > 0.5 * total_brightness:
            print(f"Array {idx}: Total brightness in A_50 exceeds 50% of total brightness.")
            break

        else:
            # Set the brightness of the brightest pixel to zero in the original array
            array[brightest_pos] = 0

    print(f"Array {idx+1}: Processed in %.3f seconds." % (time.time() - execution_time))

    print(A_50)

print("All arrays processed in %.3f seconds." % (time.time() - start_time))
