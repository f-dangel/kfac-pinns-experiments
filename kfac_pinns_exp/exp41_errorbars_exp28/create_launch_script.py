"""Create launch script to re-run best hyperparameters for different seeds."""

from argparse import ArgumentParser
from ast import literal_eval
from os import makedirs, path
from typing import Dict

from kfac_pinns_exp.exp09_reproduce_poisson2d.yaml_to_sh import QUEUE_TO_TIME
from kfac_pinns_exp.exp28_heat4d_medium.plot import DATADIR, entity, project, sweep_ids
from kfac_pinns_exp.wandb_utils import load_best_run

HEREDIR = path.dirname(path.abspath(__file__))
REPEATDIR = path.join(HEREDIR, "repeated_runs")
makedirs(REPEATDIR, exist_ok=True)

# wandb runs must be unique.
# If something goes wrong, increase this counter to create unique ids
ATTEMPT = 0


def get_commands(local_files: bool = False) -> Dict[str, Dict[str, str]]:
    """Get commands for all runs.

    Args:
        local_files: Whether to use information from locally stored runs.
            Default: `False`.

    Returns:
        A nested dictionary. The outer dictionary has sweep ids as keys. The
        inner dictionary has run ids as keys and commands as values.
    """
    commands = {sweep_id: {} for sweep_id in sweep_ids.keys()}

    for sweep_id in sweep_ids.keys():
        # load meta-data of the run
        _, df_meta = load_best_run(
            entity,
            project,
            sweep_id,
            save=False,
            update=not local_files,
            savedir=DATADIR,
        )
        df_meta = df_meta.to_dict()

        dict_config = df_meta["config"][0]
        # Loading from saved file returns a string representation that needs to
        # be evaluated into a dict.
        if local_files:
            dict_config = literal_eval(dict_config)
        run_cmd = dict_config["cmd"].split(" ")

        # drop model seed from run_cmd
        run_cmd = [arg for arg in run_cmd if "--model_seed" not in arg]

        # fill dictionary with commands to run
        run_name = df_meta["name"][0]
        for s in range(1, 11):
            run_id = f"{run_name}_model_seed_{s}_attempt_{ATTEMPT}"
            commands[sweep_id][run_id] = " ".join(
                run_cmd
                + [
                    f"--model_seed={s}",
                    f"--wandb_entity={entity}",
                    f"--wandb_project={project}",
                    f"--wandb_id={run_id}",
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

    cmds_flat = []
    for sweep_commands in get_commands(local_files=args.local_files).values():
        cmds_flat.extend(iter(sweep_commands.values()))

    jobs = [f"{cmd!r}" for cmd in cmds_flat]
    script = script.replace("JOBS_PLACEHOLDER", "\t" + "\n\t".join(jobs))
    script = script.replace("ARRAY_PLACEHOLDER", str(len(jobs) - 1))

    with open(path.join(REPEATDIR, "launch.sh"), "w") as f:
        f.write(script)
