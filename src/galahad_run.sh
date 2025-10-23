#!/bin/bash

#SBATCH --job-name=ashley-luna-mphys-rlf
#SBATCH --constraint=A100
#SBATCH --time=10-23
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=4
#SBATCH --output=/share/nas2_3/lgreen/logs/out-slurm_%j.out

pwd;

nvidia-smi
echo ">>>start"
source /share/nas2_3/lgreen/mphys-rlf/.venv/bin/activate
echo ">>>sampling"
python /share/nas2_3/lgreen/mphys-rlf/src/scripts/galahad_sample_animation.py

