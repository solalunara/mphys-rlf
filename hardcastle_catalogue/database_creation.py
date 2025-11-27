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
        self.logger.info("Loading Hardcastle catalogue from FITS file...")

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
        self.logger.info("Loading LOFAR data from H5 file...")

        # Extract the data from the h5 file
        with h5py.File(file_path, 'r') as h5file:
            lofar_images = h5file['images'][:]

        return lofar_images

    def sort_LOFAR_data(self, lofar_images):
        """
        """
        # Had this idea -- you can sort the LOFAR pixel values array by total pixel value sum, which then, by computing
        # the same for the Hardcastle catalogue, would allow the usage of e.g., a binary search to find matches faster.
        db_creator.logger.info("Sorting LOFAR data for faster matching...")
        lofar_sums = np.array([np.nansum(image) for image in lofar_images])
        sorted_indices = np.argsort(lofar_sums)
        self.logger.info(f'Lowest LOFAR pixel sum: {lofar_sums[sorted_indices[0]]}')
        self.logger.info(f'Highest LOFAR pixel sum: {lofar_sums[sorted_indices[-1]]}')
        self.logger.info(f'Median LOFAR pixel sum: {lofar_sums[sorted_indices[len(sorted_indices)//2]]}')

        # sorted_lofar_images = lofar_images[sorted_indices]

        return lofar_sums[sorted_indices], sorted_indices

    def find_dataset_matches(self, hdc_cat_items, lofar_images, lofar_sums, sorted_indices):
        self.logger.info("Finding dataset matches...")

        for idx, hdc_item in tqdm(enumerate(hdc_cat_items), desc="Finding matches"):
            hdc_sum = np.nansum(hdc_item['clipped_values'])

            # Keep track of the closest match found so far
            closest_match_idx = -1
            closest_match_diff = float('inf')

            # Use binary search to find potential matches in the sorted LOFAR sums
            left, right = 0, len(lofar_sums) - 1 # establish search bounds
            while left <= right:  # exit condition i.e., if we ever get left > right, no match has been found
                mid = (left + right) // 2  # get the middle position in the current search range
                mid_value = lofar_sums[mid] # get the LOFAR sum at that position

                # Update the closest match; this is here in case no match is found, we still have the closest one
                if abs(mid_value - hdc_sum) < closest_match_diff:
                    closest_match_diff = abs(mid_value - hdc_sum)
                    closest_match_idx = mid # note this is the sorted index and does not correspond to the original LOFAR index

                # Check for next steps
                if np.isclose(mid_value, hdc_sum, atol=1e-10): # found a match!!
                    break
                elif mid_value < hdc_sum:  # otherwise, mid value is less than target, search right half (asc so larger values are right)
                    left = mid + 1
                else: # mid value is greater than target, search left half (asc so smaller values are left)
                    right = mid - 1

            # After the binary search, we know the closest match index. Note that we are ignoring extra logic for exact
            # matches because there's no guarantee that an exact match in pixel sum means every pixel is identical.
            # Get a list of 10 indices around the closest match to check for actual pixel equality
            candidate_indices = []
            for offset in range(-5, 6):
                candidate_idx = closest_match_idx + offset # generate new index
                if 0 <= candidate_idx < len(lofar_sums): # ensure it's within bounds
                    candidate_indices.append(candidate_idx)

            # Convert sorted indices back to original LOFAR indices so we can access the corresponding images
            candidate_indices = [sorted_indices[i] for i in candidate_indices]

            # Now check these candidate indices for actual pixel equality
            matched = False
            hdc_pixels = hdc_item['clipped_values']
            for candidate_idx in candidate_indices:
                lofar_pixels = lofar_images[candidate_idx]
                try:
                    for j in range(len(lofar_pixels)):
                        if hdc_pixels[j] != lofar_pixels[j]:
                            break
                    else:
                        matched = True
                        matching_idx = candidate_idx
                        break
                except Exception as e:
                    self.logger.error(f"Error during matching for LOFAR item {candidate_idx}: {e}")

            # Exact per-pixel match found - all proof i need!
            if matched:
                self.logger.info(f'Match found for Hardcastle catalogue item index {idx} with LOFAR item index {matching_idx}.')
                return # TODO: Store the match somewhere

            # If not, we always have the closest match found via binary search
            self.logger.info(f'No exact match found for Hardcastle catalogue item index {idx} Found closest match.')
            return # TODO: Store the closest match somewhere


    def create_matching_dataset(self):
        """
        Creates a matching dataset between the Hardcastle catalogue and the given LOFAR data.
        """
        self.logger.info("Starting database creation process...")

        # Load the two datasets
        hdc_catalogue = self.load_hardcastle_catalogue()
        lofar_values = self.load_given_LOFAR_data()

        # Sort the LOFAR data for faster matching
        lofar_sums, sorted_indices = self.sort_LOFAR_data(lofar_values)

        # Find matches
        self.find_dataset_matches(hdc_catalogue, lofar_values, lofar_sums, sorted_indices)

        return

if __name__ == "__main__":
    # NOTE - PLEASE RUN hardcastle_catalogue/hardcastle_catalogue_downloader.py FIRST TO DOWNLOAD THE Hardcastle CATALOGUE FITS FILE
    # IT'S A LITTLE ANNOYING TO IMPLEMENT IN HERE BECAUSE IT USES GALAHAD DDP

    # The FITS file for the Hardcastle catalogue comes from the LoTSS-DR2 and is described by Hardcastle et al. 2023
    # It has the data with nice headers, and although it's 4.1 mil items we can narrow it done to approximately the
    # same dataset used in the paper.
    db_creator = DatabaseCreator()
    db_creator.create_matching_dataset()
