from utils.paths import LOFAR_DATA_PARENT
import urllib.request
from utils.logging import show_dl_progress
import files.paths as paths
from utils.distributed import DistributedUtils

LOFAR_DATA_PATH = LOFAR_DATA_PARENT / "LOFAR_Dataset.h5"

def download_dataset():
    paths.make_folders()
    # Check if files are present, if not download:
    files = {
        LOFAR_DATA_PATH: "https://cloud.hs.uni-hamburg.de/s/jPZdExPPmcZ48o5",
    }

    for file, link in files.items():
        if not file.exists():
            print("Downloading: ", file)
            urllib.request.urlretrieve(f"{link}/download", file, show_dl_progress)
            print("Done.")

def single_node_download_dataset():
    du = DistributedUtils()
    du.single_task_only_forcewait('download_dataset', download_dataset, 0)

if __name__ == '__main__':
    single_node_download_dataset()