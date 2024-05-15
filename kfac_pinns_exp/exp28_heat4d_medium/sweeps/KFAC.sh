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

<<<<<<<< HEAD:kfac_pinns_exp/exp28_heat4d_medium/sweeps/KFAC.sh
wandb agent --count 1 kfac-pinns/heat4d_medium/j23j5nyh
========
wandb agent --count 1 kfac-pinns/heat1d_mlp_tanh_64/vsnthmr7
>>>>>>>> master:kfac_pinns_exp/exp22_heat1d_mlp_tanh_64/sweeps/ENGD_full.sh
