#!/bin/bash
#SBATCH --partition=rtx6000
#SBATCH --qos=m5

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=1-64%17

echo "[DEBUG] Host name: " `hostname`

source  ~/anaconda3/etc/profile.d/conda.sh
conda activate kfac_pinns_exp

<<<<<<<< HEAD:kfac_pinns_exp/exp27_heat4d_small/sweeps/KFAC_auto.sh
wandb agent --count 1 kfac-pinns/heat4d_small/csxzzsr3
========
wandb agent --count 1 kfac-pinns/heat1d_mlp_tanh_64/mkh7kfhw
>>>>>>>> master:kfac_pinns_exp/exp22_heat1d_mlp_tanh_64/sweeps/ENGD_diagonal.sh
