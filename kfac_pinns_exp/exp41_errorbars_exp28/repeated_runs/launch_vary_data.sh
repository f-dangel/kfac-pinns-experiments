#!/bin/bash
#SBATCH --partition=rtx6000
#SBATCH --qos=m5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=0-89

echo "[DEBUG] Host name: " `hostname`

JOBS=(
JOBS_PLACEHOLDER
)

CMD=${JOBS[$SLURM_ARRAY_TASK_ID]}

echo Running $CMD
$CMD
