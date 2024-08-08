#!/bin/bash
#SBATCH --partition=rtx6000
#SBATCH --qos=m3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=0-8

echo "[DEBUG] Host name: " `hostname`

JOBS=(
	'python ../../train.py --optimizer=SGD --SGD_lr=0.018050148049887416 --SGD_momentum=0.99 --num_seconds=1200 --plot_solution --plot_dir=. --plot_steps 0 32494 164240 339287'
	'python ../../train.py --optimizer=Adam --Adam_lr=0.0016923389183852923 --num_seconds=1200 --plot_solution --plot_dir=. --plot_steps 0 29540 164240 320415'
	'python ../../train.py --optimizer=HessianFree --HessianFree_cg_max_iter=50 --HessianFree_curvature_opt=ggn --HessianFree_damping=100 --num_seconds=1200 --plot_solution --plot_dir=. --plot_steps 0 369 1693 3504'
	'python ../../train.py --optimizer=LBFGS --LBFGS_history_size=150 --LBFGS_lr=0.5 --num_seconds=1200 --plot_solution --plot_dir=. --plot_steps 0 2727 24414 49379'
	'python ../../train.py --optimizer=ENGD --ENGD_approximation=full --ENGD_damping=1e-10 --ENGD_ema_factor=0.3 --ENGD_initialize_to_identity --num_seconds=1200 --plot_solution --plot_dir=. --plot_steps 0 540 2727 5701'
	'python ../../train.py --optimizer=ENGD --ENGD_approximation=per_layer --ENGD_damping=0 --ENGD_ema_factor=0.9 --ENGD_initialize_to_identity --num_seconds=1200 --plot_solution --plot_dir=. --plot_steps 0 594 3000 5834'
	'python ../../train.py --optimizer=ENGD --ENGD_approximation=diagonal --ENGD_damping=0.0001 --ENGD_ema_factor=0.3 --num_seconds=1200 --plot_solution --plot_dir=. --plot_steps 0 653 3299 6714'
	'python ../../train.py --optimizer=KFAC --KFAC_damping=1.5440986717145857e-12 --KFAC_ema_factor=0.4496490467123582 --KFAC_initialize_to_identity --KFAC_momentum=0.5117574706015794 --num_seconds=1200 --plot_solution --plot_dir=. --plot_steps 0 2049 10354 20383'
	'python ../../train.py --optimizer=KFAC --KFAC_damping=1.2156399458704562e-10 --KFAC_ema_factor=0.9263313896660544 --KFAC_initialize_to_identity --KFAC_lr=auto --num_seconds=1200 --plot_solution --plot_dir=. --plot_steps 0 3299 16675 34944'
)

CMD=${JOBS[$SLURM_ARRAY_TASK_ID]}

echo Running $CMD
$CMD
