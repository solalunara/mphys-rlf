import files.dataset
import files.model
import files.paths
from utils.distributed import DistributedUtils

def prepare_for_sampling():
    files.paths.make_folders()
    files.paths.copy_config_to_sampling_dir()
    files.model.download_model()
    files.model.validate_model_in_sampling_dir()

def single_node_prepare_for_sampling():
    du = DistributedUtils()
    du.single_task_only_first('prepare_for_sampling', prepare_for_sampling, 0)

if __name__ == '__main__':
    single_node_prepare_for_sampling()