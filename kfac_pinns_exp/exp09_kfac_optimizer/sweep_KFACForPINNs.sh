#!/bin/bash
#SBATCH --qos=m3
#SBATCH --partition=rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=1-20

source  ~/anaconda3/etc/profile.d/conda.sh
conda activate kfac_pinns_exp

wandb agent --count 5 kfac-pinns/kfac-pinns-kfac_pinns_exp_exp09_kfac_optimizer/uxn8tdqg