"""Create launch script to create the solutions."""

"""Create launch script to re-run best hyperparameters for different seeds."""

from argparse import ArgumentParser
from ast import literal_eval
from os import makedirs, path
from typing import Dict

from numpy import absolute

from kfac_pinns_exp.exp09_reproduce_poisson2d.plot import (
    DATADIR,
    entity,
    project,
    sweep_ids,
)
from kfac_pinns_exp.exp09_reproduce_poisson2d.yaml_to_sh import QUEUE_TO_TIME
from kfac_pinns_exp.wandb_utils import load_best_run

HEREDIR = path.dirname(path.abspath(__file__))
VISUALIZEDIR = path.join(HEREDIR, "visualize_solution")
makedirs(VISUALIZEDIR, exist_ok=True)


def get_commands(local_files: bool = False) -> Dict[str, str]:
    """Get commands for all runs.

    Args:
        local_files: Whether to use information from locally stored runs.
            Default: `False`.

    Returns:
        A dictionary containing the commands to run for each sweep.
    """
    commands = {}

    for sweep_id, optim in sweep_ids.items():
        # load meta-data of the run
        df_history, df_meta = load_best_run(
            entity,
            project,
            sweep_id,
            save=False,
            update=not local_files,
            savedir=DATADIR,
        )
        dict_config = df_meta.to_dict()["config"][0]
        # Loading from saved file returns a string representation that needs to
        # be evaluated into a dict.
        if local_files:
            dict_config = literal_eval(dict_config)
        run_cmd = dict_config["cmd"].split(" ")

        # find out time and decrease it
        (time_arg,) = [arg for arg in run_cmd if "--num_seconds" in arg]
        time_arg = int(time_arg.split("=")[1])
        shorter_time = int(0.1 * time_arg)

        # drop time and wandb arguments
        run_cmd = [
            arg
            for arg in run_cmd
            if "--wandb" not in arg and "--num_seconds" not in arg
        ]

        # we want to visualize at initialization, around 10% of training, 50% of
        # training, and at the end of training
        visualize_ratios = [0.0005, 0.001, 0.0015, 0.002]
        visualize_steps = []
        # find the closest point
        logged_steps = df_history["step"].to_numpy()
        max_step = max(logged_steps)

        # select closest point
        for ratio in visualize_ratios:
            idx = absolute(logged_steps - ratio * max_step).argmin()
            visualize_steps.append(str(logged_steps[idx]))

        # plotting directory
        plot_dir = (
            optim.replace("(", "")
            .replace(")", "")
            .replace(" ", "_")
            .replace("*", "_auto")
        )

        commands[sweep_id] = " ".join(
            run_cmd
            + [
                f"--num_seconds={shorter_time}",
                "--plot_solution",
                f"--plot_dir={plot_dir}",
                f"--plot_steps {' '.join(visualize_steps)}",
            ]
        )

    return commands


if __name__ == "__main__":
    # write the launch script
    TEMPLATE = r"""#!/bin/bash
#SBATCH --partition=PARTITION_PLACEHOLDER
#SBATCH --qos=QOS_PLACEHOLDER
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=TIME_PLACEHOLDER
#SBATCH --mem-per-cpu=8G
#SBATCH --array=0-ARRAY_PLACEHOLDER

echo "[DEBUG] Host name: " `hostname`

JOBS=(
JOBS_PLACEHOLDER
)

CMD=${JOBS[$SLURM_ARRAY_TASK_ID]}

echo Running $CMD
$CMD
"""

    partition = "rtx6000"
    script = TEMPLATE.replace("PARTITION_PLACEHOLDER", partition)

    qos = "m5"
    time = QUEUE_TO_TIME[qos]
    script = script.replace("QOS_PLACEHOLDER", qos)
    script = script.replace("TIME_PLACEHOLDER", time)

    parser = ArgumentParser(description="Create launch script for errorbar runs.")
    parser.add_argument(
        "--local_files",
        action="store_true",
        dest="local_files",
        help="Use local files if possible.",
        default=False,
    )
    args = parser.parse_args()

    commands = get_commands(local_files=args.local_files).values()
    jobs = [f"{cmd!r}" for cmd in commands]
    script = script.replace("JOBS_PLACEHOLDER", "\t" + "\n\t".join(jobs))
    script = script.replace("ARRAY_PLACEHOLDER", str(len(jobs) - 1))

    with open(path.join(VISUALIZEDIR, "launch.sh"), "w") as f:
        f.write(script)
