#!/bin/bash -l

# Slurm parameters
#SBATCH --job-name=TRAIN5
#SBATCH --output=job_name-%j.%N.out
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1

# Activate everything you need
module load cuda/11.2
# Run your python code
python3 main.py