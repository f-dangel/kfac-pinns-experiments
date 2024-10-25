#!/bin/bash
#SBATCH --partition=rtx6000
#SBATCH --qos=m5

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=1-420%1

echo "[DEBUG] Host name: " `hostname`

wandb agent --count 1 kfac-pinns/poisson2d_mlp_tanh_256/cyswa16b
