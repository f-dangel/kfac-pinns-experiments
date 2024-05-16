#!/bin/bash
#SBATCH --partition=rtx6000
<<<<<<<< HEAD:kfac_pinns_exp/exp29_heat4d_big/sweeps/KFAC_auto.sh
#SBATCH --qos=m5

========
#SBATCH --qos=deadline
#SBATCH --account=deadline
>>>>>>>> master:kfac_pinns_exp/exp29_heat4d_big/sweeps/ENGD_diagonal.sh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=1-64%17

echo "[DEBUG] Host name: " `hostname`

source  ~/anaconda3/etc/profile.d/conda.sh
conda activate kfac_pinns_exp

<<<<<<<< HEAD:kfac_pinns_exp/exp29_heat4d_big/sweeps/KFAC_auto.sh
wandb agent --count 1 kfac-pinns/heat4d_big/4oywxhpj
========
wandb agent --count 1 kfac-pinns/heat4d_big/fjjgb3i4
>>>>>>>> master:kfac_pinns_exp/exp29_heat4d_big/sweeps/ENGD_diagonal.sh
