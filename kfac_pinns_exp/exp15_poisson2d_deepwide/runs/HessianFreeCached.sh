#!/bin/bash
#SBATCH --partition=rtx6000
#SBATCH --exclude=gpu138
#SBATCH --qos=m5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=1

echo "[DEBUG] Host name: " `hostname`

python ../../train.py --optimizer=HessianFreeCached --HessianFreeCached_cg_max_iter=350 --HessianFreeCached_damping=0.1 --boundary_condition=cos_sum --model=mlp-tanh-64-64-48-48 --num_seconds=1000 --wandb --wandb_entity=kfac-pinns --wandb_project=poisson2d_deepwide --wandb_id=HessianFreeCached
