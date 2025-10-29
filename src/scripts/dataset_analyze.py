import sys
import os
import bdsf;
from astropy.io import fits;
from astropy.io.fits import ImageHDU;
import numpy as np;
import matplotlib.pyplot as plt;
import math;
from pathlib import Path, PurePath;

# Add the src directory to Python path so we can import modules
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_dir)

import scripts.image_analyzer;
import scripts.fits_viewer;

dataset_analyzer = scripts.image_analyzer.ImageAnalyzer( "dataset" );
dataset_analyzer.AnalyzeAllFITSInInput();

fits_viewer = scripts.fits_viewer.FitsViewer();
fits_viewer.read_from_files( "fits_images/dataset/50000-60000/image50080.fits",
                             "fits_images/exported/dataset/gaus_model/50000-60000/image50080.fits" );
fits_viewer.show_image_grid( "original", "reconstruction" );