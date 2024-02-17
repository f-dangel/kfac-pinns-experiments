#!/bin/bash
#SBATCH --qos=m3
#SBATCH --partition=rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=1-25

source  ~/anaconda3/etc/profile.d/conda.sh
conda activate kfac_pinns_exp

wandb agent --count 4 kfac-pinns/kfac-pinns-kfac_pinns_exp_exp09_kfac_optimizer/pc9gt0aa
