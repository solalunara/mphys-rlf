"""
So -- it is very unclear the relationship between the LOFAR data given (LOFAR_Dataset.h5) and the actual LOFAR data
releases. From information in the paper, they refer to Hardcastle et al. 2023, which contains a catalogue of radio
sources that are "good" in the optical from LOFAR). The paper also mentions that they use only
resolved sources.

We have established from testing that the header information provided in the h5 file seems to be inaccurate - the RA
and DEC values give a galaxy with completely different pixel values from the LOFAR cutout server. This file's original
goal was to verify that we can reach the same dataset as the paper, from official resources, but we have now adapted it
to generate a new dataset with completely correct header information.

Have just now discovered that the Hardcastle catalogue does not contain pixel values. I am adding code to download the cutouts
from the LOFAR cutout server for each resolved item in the Hardcastle catalogue. Note that this is not the full catalogue.
"""

from astropy.io import fits
import numpy as np
import h5py
from tqdm import tqdm

import utils.logging
import logging
from utils.distributed import distribute
import os

num_items = 1000 # for testing purposes, limit to first 1000 items

class DatabaseCreator:
    """
    A class to create a full LOFAR database using the Hardcastle catalogue as the source, rather than the given LOFAR data.
    """

    def __init__(self):
        self.logger = utils.logging.get_logger("database creator", logging.DEBUG)

    def load_hardcastle_catalogue(self, file_path="hardcastle_catalogue/hardcastle_catalogue_with_images.fits"):
        """
        Loads the Hardcastle catalogue from a FITS file 0-clips it for matching with the present LOFAR data.

        :param file_path: The path to the Hardcastle catalogue FITS file.
        :return: A list of 0-clipped items from the catalogue.
        """
        # Get the information from the Hardcastle catalogue
        catalogue_data = []
        with fits.open(file_path) as hdul:
            # The first non-Primary table has all the header information. This'll be stored in a separate file with
            # the final matching dataset. No usage for now
            header_information = hdul[1].data

            # Remove the first two HDUs which are just Primary and the header table
            hdul = hdul[2:]

            # Extract the pixel values from each imageHDU
            for idx, hdu in enumerate(tqdm(hdul, desc="Loading Hardcastle Catalogue")):
                try:
                    catalogue_data.append({'index': idx, 'pixel_values': hdu.data})
                except Exception as e:
                    self.logger.error(f"Error loading Hardcastle catalogue item {idx}: {e}")
                    catalogue_data.append({'index':idx, 'pixel_values': np.nan})

                # TODO: For testing purposes, limit to first num_items items
                if idx >= num_items - 1:
                    break

        # 0-clip to match the LOFAR dataset's preprocessing
        self.logger.info("0-clipping the Hardcastle pixel values...")
        for i in range(len(catalogue_data)):
            try:
                if isinstance(catalogue_data[i]['pixel_values'], np.ndarray):
                    catalogue_data[i]['clipped_values'] = np.clip(catalogue_data[i]['pixel_values'], 0, None)
                else:
                    catalogue_data[i]['clipped_values'] = np.nan
            except Exception as e:
                self.logger.error(f"Error during 0-clipping for Hardcastle item {i}: {e}")

        return header_information, catalogue_data

    def load_given_LOFAR_data(self, file_path='hardcastle_catalogue/LOFAR_Dataset.h5'):
        # Extract the data from the h5 file
        with h5py.File(file_path, 'r') as h5file:
            lofar_images = h5file['images'][:]

        return lofar_images

    def create_matching_dataset(self, hdc_cat_items, lofar_item_values):
        """
        This finds matching items between the Hardcastle catalogue and the LOFAR dataset based on the pixel values of the items,
        because the header information in the LOFAR dataset is unreliable. The output is therefore every LOFAR image in the
        same order but with header information supplied from the Hardcastle catalogue.

        :param hdc_cat_items: The Hardcastle catalogue items.
        :param lofar_item_values: The LOFAR item values given by the paper.
        :return: The created dataset
        """
        num_iter = 0
        new_dataset = np.array([])

        for idx, lofar_item in tqdm(enumerate(lofar_item_values)):
            num_iter += 1

            # As we know the headers in the LOFAR data aren't relevant, the only way to match is by pixel values. The LOFAR
            # item is a 80x80 grid of pixel values; we will unravel them into a 1D array

            # Grab an Hardcastle item
            hdc_item = hdc_cat_items[idx]
            print(hdc_item)


            # Going to use the fact we know the index for 'RA' is 0 and 'DEC' is 1 in the LOFAR dataset to speed things up
            # First find wherever the RA matches, and then see if any of those also have matching DEC
            ra_match_indices = np.where(np.isclose(hdc_ra_values, lofar_ra, atol=1e-10))[0]
            dec_match_indices = np.where(np.isclose(hdc_dec_values, lofar_dec, atol=1e-10))[0]
            common_indices = np.intersect1d(ra_match_indices, dec_match_indices)
            if len(common_indices) == 0:
                continue  # No match found for RA and DEC, skip to the next Hardcastle catalogue item
            if len(common_indices) > 1:
                print(
                    f'Warning: Multiple matches found for Hardcastle catalogue item index {idx} with RA={lofar_ra} and DEC={lofar_dec}')
                break
            common_index = common_indices[0]  # this should just be a single number



            if num_iter > 1000:
                break  # Limit to first 1000 items for performance during testing



if __name__ == "__main__":
    # NOTE - PLEASE RUN hardcastle_catalogue/hardcastle_catalogue_downloader.py FIRST TO DOWNLOAD THE Hardcastle CATALOGUE FITS FILE
    # IT'S A LITTLE ANNOYING TO IMPLEMENT IN HERE BECAUSE IT USES GALAHAD DDP

    # The FITS file for the Hardcastle catalogue comes from the LoTSS-DR2 and is described by Hardcastle et al. 2023
    # It has the data with nice headers, and although it's 4.1 mil items we can narrow it done to approximately the
    # same dataset used in the paper.
    db_creator = DatabaseCreator()

    db_creator.logger.info("Starting database creation process...")

    db_creator.logger.info("Loading Hardcastle catalogue from FITS file...")
    hdc = db_creator.load_hardcastle_catalogue()

    # Unfortunately, the same is not true for the h5 file provided by the paper we're using. It's messy, with a lot of
    # headers filled with NaN values. I am doing my best to try and handle all of that but YEESH.
    db_creator.logger.info("Loading LOFAR data from H5 file...")
    lofar_values = db_creator.load_given_LOFAR_data()

    db_creator.logger.info("Creating matching dataset...")
    # db_creator.create_matching_dataset(hdc, lofar_values)
    print()