#!/bin/bash
#SBATCH --partition=rtx6000
#SBATCH --qos=deadline
#SBATCH --account=deadline
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=1-64%17

echo "[DEBUG] Host name: " `hostname`

source  ~/anaconda3/etc/profile.d/conda.sh
conda activate kfac_pinns_exp

wandb agent --count 1 kfac-pinns/heat1d_mlp_tanh_64/3mz931ye