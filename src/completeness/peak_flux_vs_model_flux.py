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

"""
Quick and dirty script to plot unscaled peak fluxes vs pybdsf-measured model fluxes for data verification
"""
if __name__ == "__main__":
    xticks = np.logspace( -5, 5, 11 )
    yticks = np.logspace( -5, 5, 11 )

    for subdir in [ utils.paths.GENERATED_SUBDIR, utils.paths.DATASET_SUBDIR ]:
        images, resid_images, model_images, model_fluxes, peak_fluxes, sigma_clipped_means, sigma_clipped_rmsds = ImageDataArrays( subdir ).get_all_arrays()
        plt.scatter( peak_fluxes, model_fluxes, label=subdir, c='b' if subdir == utils.paths.DATASET_SUBDIR else 'g', s=0.01 )

    plt.plot( xticks, yticks, c='r' )
    plt.xscale( 'log' )
    plt.yscale( 'log' )
    plt.xlabel( 'Peak Flux (mJy/pix)' )
    plt.ylabel( 'Integrated Flux (mJy)' )
    plt.grid( True )
    plt.xticks( xticks )
    plt.yticks( yticks )
    plt.legend( markerscale=100 )
    plt.savefig( 'peak_vs_model_flux.png' )
    plt.show()