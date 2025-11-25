"""
As a step of pre-processing before dataset creation, this script creates the optical catalogue by combining the header
information from the FITS file with the pixel values from the cutout files downloaded separately. This is then saved to
a new file for later use in matching to the LOFAR dataset.

For the full dataset (>300k) images, this can take ~10 minutes to the time saved by creating this file once and reading
it into many nodes on galahad is significant.
"""

"""
Please note - there is a deliberate choice in design here to not store the 'header' information from the optical catalogue
as proper FITS headers in the output file. This is because I would need to find the FITS standard for each keyword (e.g.,
Source_name breaks because it's above 8 characters. I would need to rename it to SOURCE_N or similar). There are like 
40+ key words. Rather than sinking a bunch of time into it, I have chosen to duplicate the primary table of the optical
catalogue and include a field in the header for each ImageHDU with an index linking back to the catalogue.
"""

import os
import numpy as np
from astropy.io import fits
from tqdm import tqdm

import utils.logging
import logging


num_iter = 1000  # For testing purposes, limit to first 1000 items

class OpticalCatalogueCreator:
    """
    A class to create the optical catalogue by combining header information with pixel values from cutout files.
    """

    def __init__(self):
        self.logger = utils.logging.get_logger("optical catalogue creator", logging.DEBUG)

    def load_optical_header(self, file_path="optical_catalogue/combined-release-v1.2-LM_opt_mass.fits"):
        """
        Loads the optical catalogue from a FITS file and filters for resolved items.

        :param file_path: The path to the optical catalogue FITS file.
        :return: A list of resolved items from the catalogue.
        """
        # Get the header information for the resolved items from the optical catalogue
        self.logger.info(f"Loading optical catalogue from {file_path}")
        with fits.open(file_path) as hdul:
            catalogue_data = hdul[1].data  # Assuming the data is in the first extension
            resolved_items = catalogue_data[catalogue_data['Resolved'] == True]

            #TODO: REMOVE
            resolved_items = resolved_items[:num_iter]

        # Turn resolved_items into a dictionary list for easier handling
        resolved_list = [{'header': item} for item in resolved_items]

        return resolved_list

    def load_optical_images(self, list_of_dicts, folder_path='optical_catalogue/dr2_cutouts_download/'):
        """
        Loads the optical cutout images from the specified folder.

        :param list_of_dicts: The list of dictionaries containing header information.
        :param folder_path: The path to the folder containing the cutout FITS files.
        :return: A list of optical images.
        """
        # Load the pixel values from the corresponding cutout files into memory. This is very intensive.
        self.logger.info(f"Loading optical cutout images from {folder_path}")
        for i in (tqdm(range(len(list_of_dicts)), desc="Loading pixel values")):
            cutout_file = f"{folder_path}cutout{i}.fits"

            # We run into an issue where there are hundreds of missing files because the server doesn't seem to have
            # cutouts for certain optical sources. We will log and skip these for now.
            if not os.path.exists(cutout_file):
                self.logger.info(f"Cutout file {cutout_file} does not exist. Skipping.")
                list_of_dicts[i]['pixel_values'] = np.nan
                continue

            try:
                with fits.open(cutout_file) as cutout_hdul:
                    list_of_dicts[i]['pixel_values'] = cutout_hdul[0].data
            except Exception as e:
                self.logger.error(f"Error loading cutout file {cutout_file}: {e}")

        return list_of_dicts

    def save_to_fits(self, optical_catalogue, save_path='optical_catalogue/optical_catalogue_with_images.fits'):
        """
        Saves the optical catalogue with pixel values to a FITS file.

        :param optical_catalogue: The optical catalogue with pixel values.
        :param save_path: The path to save the FITS file.
        """
        # This is a little confusing, so let me explain; I have two sources of information, the hardcastle catalogue
        # which contains the header information, and the cutout files which contain the pixel values.
        # In the hardcastle catalogue, the header is the name of the columns, and the data is the actual header info.
        self.logger.info(f"Saving optical catalogue to {save_path}")
        hdu_list = []

        # Create PrimaryHDU (empty, as we will use extensions)
        self.logger.info("Creating PrimaryHDU...")
        primary_hdu = fits.PrimaryHDU()
        hdu_list.append(primary_hdu)

        # Create BinTableHDU with the header information from the optical catalogue
        self.logger.info("Creating BinTableHDU from Hardcastle catalogue...")
        with fits.open("optical_catalogue/combined-release-v1.2-LM_opt_mass.fits") as hdul:
            resol_data = hdul[1].data[hdul[1].data['Resolved'] == True]
            hdu_list.append(fits.BinTableHDU(data=resol_data, header=hdul[1].header, name="OPTICAL_CATALOGUE"))

        # Create extension HDUs as ImageHDUs for each cutout image
        self.logger.info("Creating ImageHDUs for each cutout image...")
        for idx, item in enumerate(tqdm(optical_catalogue, desc="Creating ImageHDUs")):
            if isinstance(item['pixel_values'], float) and np.isnan(item['pixel_values']):
                continue  # Skip items with missing pixel values
            hdu = fits.ImageHDU(data=item['pixel_values'], name=f"OPTICAL_IMAGE{idx}")

            # Add WCS information to the header for pyBDSF
            hdu.header["CTYPE1"] = "RA---SIN";
            hdu.header["CTYPE2"] = "DEC--SIN";
            hdu.header["CDELT1"] = 1.5 * 0.00027778;
            hdu.header["CDELT2"] = 1.5 * 0.00027778;
            hdu.header["CUNIT1"] = "deg";
            hdu.header["CUNIT2"] = "deg";

            # Add an index so the original header information can be restored from PrimaryHDU
            hdu.header["CATIDX"] = idx
            hdu_list.append(hdu)

        hdul = fits.HDUList(hdu_list)
        hdul.writeto(save_path, overwrite=True)
        self.logger.info(f'Optical catalogue with images saved to {save_path}.')

    def create_optical_catalogue(self):
        """
        Creates the optical catalogue by loading headers and images, then combining them.
        """
        # Load the optical catalogue headers
        optical_catalogue = self.load_optical_header()
        optical_catalogue = self.load_optical_images(optical_catalogue)

        # Save the combined optical catalogue to a FITS file
        self.save_to_fits(optical_catalogue)

if __name__ == "__main__":
    occ = OpticalCatalogueCreator()
    occ.create_optical_catalogue()

    # Test loading the created catalogue
    with fits.open('optical_catalogue/optical_catalogue_with_images.fits') as hdul:
        print(hdul.info())
        print(hdul[1].data[:5])  # Print first 5 entries of the catalogue
        print(hdul[2].data)      # Print pixel values of the first image