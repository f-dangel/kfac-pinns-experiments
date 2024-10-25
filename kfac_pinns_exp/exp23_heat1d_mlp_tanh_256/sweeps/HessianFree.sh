#!/bin/bash
#SBATCH --partition=rtx6000
#SBATCH --qos=m5

#SBATCH --exclude=gpu138
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=1-64%17

echo "[DEBUG] Host name: " `hostname`

source  ~/anaconda3/etc/profile.d/conda.sh
conda activate kfac_pinns_exp

wandb agent --count 1 kfac-pinns/heat1d_mlp_tanh_256/vcu0lnb3