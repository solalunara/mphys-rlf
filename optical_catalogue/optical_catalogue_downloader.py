"""
This script downloads 314,769 files from the LOFAR cut-out server if they do not already exist. These files are needed
in the construction of a catalogue that matches the pre-processed LOFAR data given with their actual headers.
"""

import os
from pathlib import Path
import requests
import logging
import utils.logging
import sys
from astropy.io import fits
from tqdm import tqdm
import asyncio
from utils.distributed import DistributedUtils


# This is copied from stack overflow, for fast downloads using asyncio
def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
    return wrapped

class OpticalCatalogueDownloader:
    """
    This is a class to handle downloading cutouts from the LOFAR cutout server based on the optical catalogue. This is
    not a standalone step; the ~300k output files have not undergone pre-processing and further work is needed (see
    database_creation.py) to match them to the pre-processed LOFAR dataset.
    """

    def __init__(self):
        # Set up logging
        self.logger = utils.logging.get_logger("optical catalogue downloader", logging.DEBUG)

    def load_optical_catalogue(self, file_path="optical_catalogue/combined-release-v1.2-LM_opt_mass.fits"):
        """
        Loads the optical catalogue from a FITS file and filters for resolved items. This turns the ~4.1mil items from the
        LoTSS-DR2 release w/ optical sources to 314,769 values. Note that this does not get pixel values; that is what this
        whole script is for.

        :param file_path: The path to the optical catalogue FITS file.
        :return: A list of resolved items from the catalogue.
        """
        try:
            with fits.open(file_path) as hdul:
                catalogue_data = hdul[1].data
        except Exception as e:
            print(f"Error loading FITS file: {e}")
            return []

        # Get the headers of resolved sources
        resolved_items = catalogue_data[catalogue_data['Resolved'] == True]

        return resolved_items

    def get_optical_positions(self, optical_catalogue):
        """
        Extracts the optical positions (RA, DEC) from the resolved items in the optical catalogue.

        :param optical_catalogue: The list of resolved items from the optical catalogue.
        :return: A list of tuples containing (RA, DEC) for each resolved item.
        """
        positions = []
        for item in optical_catalogue:
            ra = item['RA']
            dec = item['DEC']
            positions.append((ra, dec))
        return positions

    # NOTE-THIS COMES FROM THE LOFAR API
    @background
    def get_cutout(self, outfile, pos, size=2, low=False, dr3=False, auth=None):
        '''Get a cutout at position pos with size size arcmin. If low is
        True, get the 20-arcsec cutout, else get the 6-arcsec one. If dr3
        is true, try to access the DR3 data instead. Save to outfile.

        '''
        base = 'dr3' if dr3 else 'dr2'
        url = 'https://lofar-surveys.org/'
        if low:
            page = base + '-low-cutout.fits'
        else:
            page = base + '-cutout.fits'

        self.logger.debug(f'Trying {url + page}?pos={pos}&size={size}')
        r = requests.get(url + page, params={'pos': pos, 'size': size}, auth=auth, stream=True)
        self.logger.debug(f'received response code {r.status_code} and content type {r.headers["content-type"]}')
        if r.status_code != 200:
            raise RuntimeError('Status code %i returned' % r.status_code)
        if r.headers['content-type'] != 'application/fits':
            raise RuntimeError('Server did not return FITS file, probably no coverage of this area')

        with open(outfile, 'wb') as o:
            o.write(r.content)
            r.close()

    def download_all_cutouts(self):
        """
        Downloads all cutouts from the LOFAR cutout server based on the optical catalogue positions.
        """
        # Prepare for distributed processing on galahad
        du = DistributedUtils()
        task_count = du.get_task_count()
        task_id = du.get_task_id()
        self.logger.debug(f'Task ID: {task_id}, Task Count: {task_count}')

        optical_catalogue = self.load_optical_catalogue()
        opt_positions = self.get_optical_positions(optical_catalogue)

        # Check if target directory exists, create if not
        target_directory = "optical_catalogue/dr2_cutouts_download"
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        # Create a list of image numbers corresponding to the positions
        image_nums = list(range(len(opt_positions)))

        # Prepare for simultaneous downloads using multiple CPUs based on binning
        n_cpus = os.environ.get("N_CPUS", 1)
        if isinstance(n_cpus, str):
            n_cpus = int(n_cpus)
        self.logger.info("Using %i cpu" + ("s" if n_cpus != 1 else ""), n_cpus)

        # distribute across multiple tasks
        n_files = len(image_nums)
        bin_start = du.get_bin_start(n_files)
        bin_end = du.get_bin_end(n_files)
        image_nums = image_nums[bin_start:bin_end]  # each node only interacts with its own bin

        for i in tqdm(image_nums):
            # get the RA and DEC for this image number
            ra, dec = opt_positions[i]

            # check if file exists and don't download if so
            if os.path.exists(f"optical_catalogue/dr2_cutouts_download/cutout{i}.fits"):
                continue
            print(f'Downloading image {i} for RA={ra}, DEC={dec} degrees')
            self.get_cutout(f"optical_catalogue/dr2_cutouts_download/cutout{i}.fits", f"{ra} {dec}")


if __name__ == "__main__":
    # Prepare for distributed processing on galahad
    # Get a list of positions from the optical catalogue
    # Running into a possible problem where galahad nodes don't want to read the same file; will force a single node
    # to perform this, and then copy the results to all nodes.

    # file_path = Path("optical_catalogue/combined-release-v1.2-LM_opt_mass.fits")
    # du.copy_file_for_multiple_nodes(file_path)
    # du.single_task_only_forcewait('load_optical_catalogue', load_optical_catalogue, 0)
    downloader = OpticalCatalogueDownloader()
    downloader.download_all_cutouts()

