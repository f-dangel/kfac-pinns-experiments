# Launch all sweeps using the sbatch command
cd sweeps/

# launch each script
sbatch KFAC.sh
sbatch KFAC_auto.sh
sbatch SGD.sh
sbatch Adam.sh
sbatch LBFGS.sh
sbatch HessianFree.sh
