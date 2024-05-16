#!/bin/bash
#SBATCH --partition=rtx6000
#SBATCH --qos=deadline
#SBATCH --account=deadline
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=1-64%17

echo "[DEBUG] Host name: " `hostname`

source  ~/anaconda3/etc/profile.d/conda.sh
conda activate kfac_pinns_exp

<<<<<<<< HEAD:kfac_pinns_exp/exp29_heat4d_big/sweeps/Adam.sh
wandb agent --count 1 kfac-pinns/heat4d_big/1bbrvf9u
========
wandb agent --count 1 kfac-pinns/heat4d_big/hsbs4ryg
>>>>>>>> master:kfac_pinns_exp/exp29_heat4d_big/sweeps/HessianFree.sh
