# This file has been created by Ashley and Luna. It defines the parameters used in the program,
# allowing easy tweaking for running on different devices

# Default bins sizes to sort data from the dataset into when converting h5 to fits
BINS_ARRAY = [ 10000 ];

# Default parameters for fits file generation are placed here for ease of maintenance
FITS_SAMPLING_ARGS = dict(
    batch_size = 10,
    n_samples = 4,
    timesteps = 25,
    bin_size = 10000,
    initial_count = -1,
    use_cpu=True
);

# Default number of fits images to convert
LOFAR_FITS_COUNT_CUTOFF = 99;