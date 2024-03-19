#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=1-10

source  ~/anaconda3/etc/profile.d/conda.sh
conda activate kfac_pinns_exp

wandb agent --count 5 kfac-pinns/poisson2d/yzdbc4h3