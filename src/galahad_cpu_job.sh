#!/bin/bash

#SBATCH --job-name=ashley-luna-mphys-rlf
#SBATCH --constraint=A100
#SBATCH --time=1-23
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=/share/nas2_3/lgreen/logs/out-slurm_%j.out
#SBATCH -c 24
#SBATCH --no-requeue

pwd;

nvidia-smi
echo ">>>start"
source /share/nas2_3/lgreen/mphys-rlf/.venv/bin/activate
echo ">>>sampling"
N_CPUS=24 python /share/nas2_3/lgreen/mphys-rlf/src/scripts/image_analyzer.py

