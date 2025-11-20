"""
Module to verify the completeness of downloaded cutout files from an optical catalogue. This is done separately to
the downloader script to avoid the issue of nodes attempting to verify and re-download all images before every image
has properly been downloaded.
"""

import os
import logging
import utils.logging

from optical_catalogue_downloader import OpticalCatalogueDownloader

def verify_downloads(downloader, optical_catalogue, download_path="optical_catalogue/dr2_cutouts_download"):
    """
    Verifies that all cutout files have been downloaded.

    :param optical_catalogue: The optical catalogue for the downloaded images.
    :param download_path: The path where the cutout images are stored.
    """
    total_images = len(optical_catalogue)
    opt_positions = downloader.get_optical_positions(optical_catalogue)
    files_to_redownload = []

    # Check for missing images
    for i in range(total_images):
        file_path = os.path.join(download_path, f"cutout{i}.fits")

        # Check for missing images
        if not os.path.exists(file_path):
            logger.error(f'Missing cutout file: {file_path}.')
            files_to_redownload.append(i)
            continue

        # Check that each image can be loaded and is therefore not corrupted
        try:
            from astropy.io import fits
            with fits.open(file_path) as hdul:
                _ = hdul[0].data  # Attempt to read the data
        except Exception as e:
            logger.error(f'Corrupted or empty cutout file: {file_path}.')
            files_to_redownload.append(i)
            continue

    # Redownload any files if necessary
    if files_to_redownload:
        logger.info(f'Total missing cutout files: {len(files_to_redownload)}. Redownloading...')
        for i in files_to_redownload:
            ra, dec = opt_positions[i]
            try:
                logger.info(f'Redownloading cutout for image {i} (RA={ra}, DEC={dec})...')
                downloader.get_cutout(os.path.join(download_path, f"cutout{i}.fits"), f"{ra} {dec}")
            except Exception as e:
                logger.error(f"Error redownloading cutout for image {i} (RA={ra}, DEC={dec}): {e}")
    else:
        logger.info('All cutout files are present.')


if __name__ == "__main__":
    # Prepare logger
    logger = utils.logging.get_logger("download verifier", logging.INFO)

    logger.info('Loading optical catalogue for verification...')
    downloader = OpticalCatalogueDownloader()
    optical_catalogue = downloader.load_optical_catalogue()
    logger.info(f'Loaded optical catalogue with {len(optical_catalogue)} resolved items.')

    logger.info('Starting verification of downloaded cutouts...')
    verify_downloads(downloader, optical_catalogue)