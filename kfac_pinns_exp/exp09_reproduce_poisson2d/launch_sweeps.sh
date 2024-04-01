# Launch all sweeps using the sbatch command
cd sweeps/
sbatch SGD.sh
sbatch Adam.sh
sbatch LBFGS.sh
sbatch HessianFree.sh
sbatch ENGD_diagonal.sh
sbatch ENGD_full.sh
sbatch ENGD_per_layer.sh
sbatch KFAC.sh
