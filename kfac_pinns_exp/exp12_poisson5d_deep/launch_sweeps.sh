# Launch all sweeps using the sbatch command
cd sweeps/
sbatch SGD.sh
sbatch Adam.sh
sbatch KFAC.sh
sbatch KFAC_empirical.sh
sbatch KFAC_forward_only.sh
sbatch LBFGS.sh
sbatch HessianFree.sh
sbatch ENGD_diagonal.sh
sbatch ENGD_full.sh
sbatch ENGD_per_layer.sh
