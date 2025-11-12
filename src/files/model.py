from utils.paths import PRETRAINED_PARENT, LOFAR_MODEL_PARENT, LOFAR_MODEL_NAME
import urllib.request
from utils.logging import show_dl_progress
import shutil
import files.paths as paths
from utils.distributed import DistributedUtils

def download_model():
    paths.make_folders()
    # Check if files are present, if not download:
    files = {
        PRETRAINED_PARENT
        / "parameters_LOFAR_model.pt": "https://cloud.hs.uni-hamburg.de/s/KTAFWFnLByMgNRn",
        PRETRAINED_PARENT
        / "parameters_FIRST_model.pt": "https://cloud.hs.uni-hamburg.de/s/xs7bbt99AMFf8gP",
    }

    for file, link in files.items():
        if not file.exists():
            print("Downloading: ", file)
            urllib.request.urlretrieve(f"{link}/download", file, show_dl_progress)
            print("Done.")

def validate_model_in_sampling_dir():
    file = LOFAR_MODEL_PARENT / f"parameters_{LOFAR_MODEL_NAME}.pt"
    if not file.exists():
        download_model()
        shutil.copyfile(PRETRAINED_PARENT / "parameters_LOFAR_model.pt", LOFAR_MODEL_PARENT / f"parameters_{LOFAR_MODEL_NAME}.pt")

def single_node_validate_model_in_sampling_dir():
    du = DistributedUtils()
    du.single_task_only_forcewait('validate_model_in_sampling_dir', validate_model_in_sampling_dir, 0)


if __name__ == '__main__':
    single_node_validate_model_in_sampling_dir();