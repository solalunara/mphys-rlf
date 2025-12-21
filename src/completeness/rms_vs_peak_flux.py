from pybdsf_analysis.recursive_file_analyzer import RecursiveFileAnalyzer
from pybdsf_analysis.log_analyzer import LogAnalyzer
import utils.paths
import pybdsf_analysis.log_analyzer as la
import pybdsf_analysis.recursive_file_analyzer as rfa
import numpy as np
import h5py
from sklearn.preprocessing import PowerTransformer
from files.dataset import LOFAR_DATA_PATH
import matplotlib.pyplot as plt
from completeness.img_data_arrays import ImageDataArrays
import pybdsf_analysis.pybdsf_run_analysis

"""
Quick and dirty script to plot unscaled peak fluxes vs pybdsf-measured model fluxes for data verification
"""
if __name__ == "__main__":
    #pybdsf_analysis.pybdsf_run_analysis.analyze_everything()
    for subdir in [ utils.paths.GENERATED_SUBDIR, utils.paths.DATASET_SUBDIR ]:
        images, resid_images, model_images, model_fluxes, peak_fluxes, sigma_clipped_means, sigma_clipped_rmsds = ImageDataArrays( subdir ).get_all_arrays()
        plt.scatter( peak_fluxes, np.std( resid_images, axis=(1,2) ), label=subdir, c='b' if subdir == utils.paths.DATASET_SUBDIR else 'g', s=0.01 )

    plt.xscale( 'log' )
    plt.yscale( 'log' )
    plt.xlabel( 'Peak Flux (mJy/pix)' )
    plt.ylabel( 'RMS (mJy)' )
    plt.grid( True )
    plt.legend( markerscale=100 )
    plt.savefig( 'peak_vs_model_flux.png' )
    plt.show()