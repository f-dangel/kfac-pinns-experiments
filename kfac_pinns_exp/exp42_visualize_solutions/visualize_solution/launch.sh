#!/bin/bash
#SBATCH --partition=rtx6000
#SBATCH --qos=m5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=0-17

echo "[DEBUG] Host name: " `hostname`

JOBS=(
	'python ../../train.py --optimizer=SGD --SGD_lr=0.018050148049887416 --SGD_momentum=0.99 --num_seconds=200 --save_checkpoints --checkpoint_dir=SGD --checkpoint_steps 0 335 3299 32494 && python ../../plot_solution.py --checkpoint_dir=SGD --plot_dir=SGD --disable_tex'
	'python ../../train.py --optimizer=Adam --Adam_lr=0.0016923389183852923 --num_seconds=200 --save_checkpoints --checkpoint_dir=Adam --checkpoint_steps 0 335 3299 32494 && python ../../plot_solution.py --checkpoint_dir=Adam --plot_dir=Adam --disable_tex'
	'python ../../train.py --optimizer=HessianFree --HessianFree_cg_max_iter=50 --HessianFree_curvature_opt=ggn --HessianFree_damping=100 --num_seconds=200 --save_checkpoints --checkpoint_dir=Hessian-free --checkpoint_steps 0 4 35 335 && python ../../plot_solution.py --checkpoint_dir=Hessian-free --plot_dir=Hessian-free --disable_tex'
	'python ../../train.py --optimizer=LBFGS --LBFGS_history_size=150 --LBFGS_lr=0.5 --num_seconds=200 --save_checkpoints --checkpoint_dir=LBFGS --checkpoint_steps 0 50 491 4831 && python ../../plot_solution.py --checkpoint_dir=LBFGS --plot_dir=LBFGS --disable_tex'
	'python ../../train.py --optimizer=ENGD --ENGD_approximation=full --ENGD_damping=1e-10 --ENGD_ema_factor=0.3 --ENGD_initialize_to_identity --num_seconds=200 --save_checkpoints --checkpoint_dir=ENGD_full --checkpoint_steps 0 6 55 594 && python ../../plot_solution.py --checkpoint_dir=ENGD_full --plot_dir=ENGD_full --disable_tex'
	'python ../../train.py --optimizer=ENGD --ENGD_approximation=per_layer --ENGD_damping=0 --ENGD_ema_factor=0.9 --ENGD_initialize_to_identity --num_seconds=200 --save_checkpoints --checkpoint_dir=ENGD_layer-wise --checkpoint_steps 0 6 61 594 && python ../../plot_solution.py --checkpoint_dir=ENGD_layer-wise --plot_dir=ENGD_layer-wise --disable_tex'
	'python ../../train.py --optimizer=ENGD --ENGD_approximation=diagonal --ENGD_damping=0.0001 --ENGD_ema_factor=0.3 --num_seconds=200 --save_checkpoints --checkpoint_dir=ENGD_diagonal --checkpoint_steps 0 7 67 653 && python ../../plot_solution.py --checkpoint_dir=ENGD_diagonal --plot_dir=ENGD_diagonal --disable_tex'
	'python ../../train.py --optimizer=KFAC --KFAC_damping=1.5440986717145857e-12 --KFAC_ema_factor=0.4496490467123582 --KFAC_initialize_to_identity --KFAC_momentum=0.5117574706015794 --num_seconds=200 --save_checkpoints --checkpoint_dir=KFAC --checkpoint_steps 0 20 208 2049 && python ../../plot_solution.py --checkpoint_dir=KFAC --plot_dir=KFAC --disable_tex'
	'python ../../train.py --optimizer=KFAC --KFAC_damping=1.2156399458704562e-10 --KFAC_ema_factor=0.9263313896660544 --KFAC_initialize_to_identity --KFAC_lr=auto --num_seconds=200 --save_checkpoints --checkpoint_dir=KFAC_auto --checkpoint_steps 0 35 335 3629 && python ../../plot_solution.py --checkpoint_dir=KFAC_auto --plot_dir=KFAC_auto --disable_tex'
	'python ../../train.py --optimizer=SGD --SGD_lr=0.017527517052736283 --SGD_momentum=0.99 --dim_Omega=1 --equation=heat --num_seconds=200 --save_checkpoints --checkpoint_dir=SGD --checkpoint_steps 0 277 3000 29540 && python ../../plot_solution.py --checkpoint_dir=SGD --plot_dir=SGD --disable_tex'
	'python ../../train.py --optimizer=Adam --Adam_lr=0.0008629006288964788 --dim_Omega=1 --equation=heat --num_seconds=200 --save_checkpoints --checkpoint_dir=Adam --checkpoint_steps 0 277 2727 26855 && python ../../plot_solution.py --checkpoint_dir=Adam --plot_dir=Adam --disable_tex'
	'python ../../train.py --optimizer=HessianFree --HessianFree_cg_max_iter=250 --HessianFree_curvature_opt=ggn --HessianFree_damping=0.0001 --dim_Omega=1 --equation=heat --num_seconds=200 --save_checkpoints --checkpoint_dir=Hessian-free --checkpoint_steps 0 1 11 107 && python ../../plot_solution.py --checkpoint_dir=Hessian-free --plot_dir=Hessian-free --disable_tex'
	'python ../../train.py --optimizer=LBFGS --LBFGS_history_size=125 --LBFGS_lr=0.1 --dim_Omega=1 --equation=heat --num_seconds=200 --save_checkpoints --checkpoint_dir=LBFGS --checkpoint_steps 0 55 594 5845 && python ../../plot_solution.py --checkpoint_dir=LBFGS --plot_dir=LBFGS --disable_tex'
	'python ../../train.py --optimizer=ENGD --ENGD_approximation=full --ENGD_damping=1e-12 --ENGD_ema_factor=0.9 --dim_Omega=1 --equation=heat --num_seconds=200 --save_checkpoints --checkpoint_dir=ENGD_full --checkpoint_steps 0 4 42 446 && python ../../plot_solution.py --checkpoint_dir=ENGD_full --plot_dir=ENGD_full --disable_tex'
	'python ../../train.py --optimizer=ENGD --ENGD_approximation=per_layer --ENGD_damping=1e-10 --ENGD_ema_factor=0.3 --dim_Omega=1 --equation=heat --num_seconds=200 --save_checkpoints --checkpoint_dir=ENGD_layer-wise --checkpoint_steps 0 4 42 446 && python ../../plot_solution.py --checkpoint_dir=ENGD_layer-wise --plot_dir=ENGD_layer-wise --disable_tex'
	'python ../../train.py --optimizer=ENGD --ENGD_approximation=diagonal --ENGD_damping=1e-06 --ENGD_ema_factor=0.99 --dim_Omega=1 --equation=heat --num_seconds=200 --save_checkpoints --checkpoint_dir=ENGD_diagonal --checkpoint_steps 0 5 50 491 && python ../../plot_solution.py --checkpoint_dir=ENGD_diagonal --plot_dir=ENGD_diagonal --disable_tex'
	'python ../../train.py --optimizer=KFAC --KFAC_damping=1.2737541166667861e-08 --KFAC_ema_factor=0.3611723814450289 --KFAC_initialize_to_identity --KFAC_momentum=0.7562617370720113 --dim_Omega=1 --equation=heat --num_seconds=200 --save_checkpoints --checkpoint_dir=KFAC --checkpoint_steps 0 14 143 1400 && python ../../plot_solution.py --checkpoint_dir=KFAC --plot_dir=KFAC --disable_tex'
	'python ../../train.py --optimizer=KFAC --KFAC_damping=1.9684273404512036e-09 --KFAC_ema_factor=0.9703637899133728 --KFAC_initialize_to_identity --KFAC_lr=auto --dim_Omega=1 --equation=heat --num_seconds=200 --save_checkpoints --checkpoint_dir=KFAC_auto --checkpoint_steps 0 34 335 3299 && python ../../plot_solution.py --checkpoint_dir=KFAC_auto --plot_dir=KFAC_auto --disable_tex'
)

CMD=${JOBS[$SLURM_ARRAY_TASK_ID]}

echo Running $CMD
$CMD
