# Overview
This repository is a fork of [tmartinezML/LOFAR-Diffusion](https://github.com/tmartinezML/LOFAR-Diffusion/) by Ashley Parr and Luna Greenberg for the project "Generative AI for Calculating the Completeness of
Resolved LoTSS DR2 Radio Sources."


# Installation

1. Clone the repository to your local machine:
```bash
git clone https://github.com/tmartinezML/LOFAR-Diffusion.git
cd LOFAR-Diffusion
```

2. Create a new virtual environment. Make sure to use python 3.11 or 3.10. Install the required packages to your environment:
```bash
pip install -r src/requirements.txt
```
__Important__: If you use a python version >= 3.12, you will run into compatibility issues with the required packages!

Also, if running on SLURM or some other distributed system it may be useful to use the modified functions.py file in PyBDSF included in this repository

3. By default, data will be stored in the repository directory. You can change this by setting the STORAGE_PARENT path in  src/utils/paths.py

4. Run any file which generates a figure, such as src/scripts/flux_vs_residual.py:
```bash
python src/scripts/flux_vs_residual.py
```
This will set up the correct folder structure and download trained model weights, as well as the LOFAR training set, sample the model, and run PyBDSF on the dataset and generated images. The FIRST dataset will be downloaded when it is first initialized.

# Usage

Training is executed with the script training/train.py. Settings are defined with a json configuration file that has to be specified inside the script. Example configs are given in model/configs. Also, the training dataset is specified in the script. Different options are defined in data/datasets.py.

Parameters used in the program are defined in src/utils/parameters.py, and can be overridden in an untracked file src/utils/parameters_local_override.py if desired.
