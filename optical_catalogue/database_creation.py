"""
So -- it is very unclear the relationship between the LOFAR data given (LOFAR_Dataset.h5) and the actual LOFAR data
releases. From information in the paper, they refer to Hardcastle et al. 2023, which contains an opticl catalogue (i.e.,
a catalogue of radio sources that are "good" in the optical from LOFAR). The paper also mentions that they use only
resolved sources.

We have established from testing that the header information provided in the h5 file seems to be inaccurate - the RA
and DEC values give a galaxy with completely different pixel values from the LOFAR cutout server. This file's original
goal was to verify that we can reach the same dataset as the paper, from official resources, but we have now adapted it
to generate a new dataset with completely correct header information.

Have just now discovered that the optical catalogue does not contain pixel values. I am adding code to download the cutouts
from the LOFAR cutout server for each resolved item in the optical catalogue. Note that this is not the full catalogue.
"""

from astropy.io import fits
import numpy as np
import h5py
from tqdm import tqdm

import utils.logging
import logging

class DatabaseCreator:
    """
    A class to create a full
    """

    def __init__(self):
        self.logger = utils.logging.get_logger("database creator", logging.DEBUG)

    def load_optical_catalogue(self, file_path="combined-release-v1.2-LM_opt_mass.fits"):
        """
        Loads the optical catalogue from a FITS file and filters for resolved items.

        :param file_path: The path to the optical catalogue FITS file.
        :return: A list of resolved items from the catalogue.
        """
        try:
            with fits.open(file_path) as hdul:
                # catalogue_images = hdul[2].data
                catalogue_data = hdul[1].data
                print(hdul.info())
        except Exception as e:
            print(f"Error loading FITS file: {e}")
            return []

        # Extract the Resolved flag
        resolved_flags = catalogue_data['Resolved']

        # Get the indices of resolved items
        resolved_indices = np.where(resolved_flags == True)[0]

        # Get the resolved images using these indices
        print(catalogue_data)
        # resolved_images = catalogue_images[resolved_indices]

        return resolved_images


    def load_given_LOFAR_data(self, file_path='LOFAR_Dataset.h5'):
        # Extract the data from the h5 file
        with h5py.File(file_path, 'r') as h5file:
            lofar_images = h5file['images'][:]
            print(len(lofar_images))

            return lofar_images

    def create_matching_dataset(self, opt_cat_items, lofar_item_values):
        """
        This finds matching items between the optical catalogue and the LOFAR dataset based on the pixel values of the items,
        because the header information in the LOFAR dataset is unreliable. The output is therefore every LOFAR image in the
        same order but with header information supplied from the optical catalogue.

        :param opt_cat_items: The optical catalogue items.
        :param lofar_item_values: The LOFAR item values given by the paper.
        :return: The created dataset
        """
        num_iter = 0
        new_dataset = np.array([])

        for idx, lofar_item in tqdm(enumerate(lofar_item_values)):
            num_iter += 1

            # As we know the headers in the LOFAR data aren't relevant, the only way to match is by pixel values. The LOFAR
            # item is a 80x80 grid of pixel values; we will unravel them into a 1D array

            # Grab an optical item
            opt_item = opt_cat_items[idx]
            print(opt_item)


            # Going to use the fact we know the index for 'RA' is 0 and 'DEC' is 1 in the LOFAR dataset to speed things up
            # First find wherever the RA matches, and then see if any of those also have matching DEC
            ra_match_indices = np.where(np.isclose(opt_ra_values, lofar_ra, atol=1e-10))[0]
            dec_match_indices = np.where(np.isclose(opt_dec_values, lofar_dec, atol=1e-10))[0]
            common_indices = np.intersect1d(ra_match_indices, dec_match_indices)
            if len(common_indices) == 0:
                continue  # No match found for RA and DEC, skip to the next optical catalogue item
            if len(common_indices) > 1:
                print(
                    f'Warning: Multiple matches found for optical catalogue item index {idx} with RA={lofar_ra} and DEC={lofar_dec}')
                break
            common_index = common_indices[0]  # this should just be a single number



            if num_iter > 1000:
                break  # Limit to first 1000 items for performance during testing



if __name__ == "__main__":
    # NOTE - PLEASE RUN optical_catalogue/optical_catalogue_downloader.py FIRST TO DOWNLOAD THE OPTICAL CATALOGUE FITS FILE
    # IT'S A LITTLE ANNOYING TO IMPLEMENT IN HERE BECAUSE IT USES GALAHAD DDP

    # The FITS file for the optical catalogue comes from the LoTSS-DR2 and is described by Hardcastle et al. 2023
    # It has the data with nice headers, and although it's 4.1 mil items we can narrow it done to approximately the
    # same dataset used in the paper.
    db_creator = DatabaseCreator()

    print("Loading optical catalogue from FITS file...")
    resolved_items = db_creator.load_optical_catalogue()

    # Unfortunately, the same is not true for the h5 file provided by the paper we're using. It's messy, with a lot of
    # headers filled with NaN values. I am doing my best to try and handle all of that but YEESH.
    print("Loading LOFAR data from H5 file...")
    lofar_values = db_creator.load_given_LOFAR_data()

    print("Creating matching dataset...")
    db_creator.create_matching_dataset(resolved_items, lofar_values)
    print()