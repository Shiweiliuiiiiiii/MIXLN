#!/bin/bash
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gpus=4
#SBATCH -t 1-15:00:00 
#SBATCH --cpus-per-task=36
#SBATCH -o mix_norm.out

source /home/sliu/anaconda3/etc/profile.d/conda.sh
source activate galore_3.11

bash run_1b.sh post_pre 5e-4