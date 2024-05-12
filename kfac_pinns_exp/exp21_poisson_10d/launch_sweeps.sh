# Launch all sweeps using the sbatch command
cd sweeps/

# launch each script
sbatch KFAC.sh
sbatch KFAC_empirical.sh
sbatch KFAC_forward_only.sh
sbatch SGD.sh
sbatch Adam.sh
sbatch LBFGS.sh
sbatch HessianFree.sh
sbatch KFAC_auto.sh
sbatch KFAC_empirical_auto.sh
sbatch KFAC_forward_only_auto.sh
