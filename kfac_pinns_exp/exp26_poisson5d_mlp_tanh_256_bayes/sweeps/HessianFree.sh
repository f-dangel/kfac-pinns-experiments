#!/bin/bash
#SBATCH --partition=rtx6000
#SBATCH --qos=m3
#SBATCH --exclude=gpu138

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=1-250%1

echo "[DEBUG] Host name: " `hostname`

wandb agent --count 1 kfac-pinns/poisson5d_mlp_tanh_256_bayes/jjxhro1a
