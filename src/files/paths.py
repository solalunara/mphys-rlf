import utils.paths as pth;
import shutil;
from utils.distributed import DistributedUtils;

def make_folders():
    # Create folders and symlinks
    for p in [pth.MODEL_PARENT, pth.ANALYSIS_PARENT, pth.IMG_DATA_PARENT, pth.FITS_PARENT, pth.PYBDSF_PARENT, pth.NP_ARRAY_PARENT, pth.FLAGS_PARENT]:
        # Make folder if it doesn't exist
        if not p.exists():
            p.mkdir()

        # Create symlink if necessary
        if not pth.STORAGE_PARENT == pth.BASE_PARENT:
            symlink = pth.BASE_PARENT / p.name
            if not symlink.exists():
                symlink.symlink_to(p)
            else:
                assert (
                    symlink.resolve() == p
                ), f"Broken folder structure: Symlink {symlink} points to {symlink.resolve()}."


    for f in [pth.LOFAR_DATA_PARENT, 
              pth.FIRST_DATA_PARENT, 
              pth.PRETRAINED_PARENT, 
              pth.LOFAR_MODEL_PARENT,
              pth.MOSAIC_DIR,
              pth.CUTOUTS_DIR]:
        if not f.exists():
            f.mkdir()

    for f in [pth.FITS_PARENT,
              pth.PYBDSF_CATALOG_PARENT,
              pth.PYBDSF_EXPORT_IMAGE_PARENT,
              pth.PYBDSF_LOG_PARENT,
              pth.NP_ARRAY_PARENT]:
        if not f.exists():
            f.mkdir()
        for g in [pth.DATASET_SUBDIR, pth.GENERATED_SUBDIR]:
            # ignore FITS_PARENT/DATASET_SUBDIR because its existence can be used to determine if the dataset should be recreated
            if not (f/g).exists() and not ( f == pth.FITS_PARENT and g == pth.DATASET_SUBDIR ):
                (f/g).mkdir()


def copy_config_to_sampling_dir():
    # Copy sample lofar config to sampling directory
    LOFAR_SAMPLING_CONFIG_PATH = pth.LOFAR_MODEL_PARENT / f"config_{pth.LOFAR_MODEL_NAME}.json"
    if not LOFAR_SAMPLING_CONFIG_PATH.exists():
        shutil.copy(pth.CONFIG_PARENT / "LOFAR_Model.json", LOFAR_SAMPLING_CONFIG_PATH)

def make_folders_and_copy_config():
    make_folders()
    copy_config_to_sampling_dir()

def single_node_prepare_folders():
    du = DistributedUtils()
    du.single_task_only_forcewait('make_folders_and_copy_config', make_folders_and_copy_config, 0)

if __name__ == '__main__':
    single_node_prepare_folders()