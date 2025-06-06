#!/bin/bash
#SBATCH --partition=rtx6000
#SBATCH --qos=m4

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=1-100%1

echo "[DEBUG] Host name: " `hostname`

source  ~/anaconda3/etc/profile.d/conda.sh
conda activate kfac_pinns_exp

wandb agent --count 1 kfac-pinns/heat4d_mlp_tanh_256_bayes/ml685rb5