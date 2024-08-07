"""Create launch script to re-run best hyperparameters for different seeds."""

from os import makedirs, path

from kfac_pinns_exp.exp09_reproduce_poisson2d.yaml_to_sh import QUEUE_TO_TIME
from kfac_pinns_exp.exp28_heat4d_medium.plot import DATADIR, entity, project, sweep_ids
from kfac_pinns_exp.wandb_utils import load_best_run

HEREDIR = path.dirname(path.abspath(__file__))
REPEATDIR = path.join(HEREDIR, "repeated_runs")
makedirs(REPEATDIR, exist_ok=True)

# wandb runs must be unique.
# If something goes wrong, increase this counter to create unique ids
ATTEMPT = 0

VARY_MODEL_COMMANDS = {sweep_id: {} for sweep_id in sweep_ids.keys()}
VARY_DATA_COMMANDS = {sweep_id: {} for sweep_id in sweep_ids.keys()}

for sweep_id, label in sweep_ids.items():
    # load meta-data of the run
    _, df_meta = load_best_run(
        entity,
        project,
        sweep_id,
        save=True,
        update=True,
        savedir=DATADIR,
    )
    df_meta = df_meta.to_dict()
    run_name = df_meta["name"][0]
    run_cmd = df_meta["config"][0]["cmd"].split(" ")

    # drop model and data seeds from run_cmd
    run_cmd = [
        arg for arg in run_cmd if "--model_seed" not in arg and "--data_seed" not in arg
    ]

    # fill dictionaries with commands to run
    for s in range(1, 11):
        run_id = f"{run_name}_model_seed_{s}_attempt_{ATTEMPT}"
        VARY_MODEL_COMMANDS[sweep_id][run_id] = " ".join(
            run_cmd
            + [
                f"--model_seed={s}",
                "--data_seed=0",
                f"--wandb_entity={entity}",
                f"--wandb_project={project}",
                f"--wandb_id={run_id}",
            ]
        )

    for s in range(10):
        run_id = f"{run_name}_data_seed_{s}_attempt_{ATTEMPT}"
        VARY_DATA_COMMANDS[sweep_id][run_id] = " ".join(
            run_cmd
            + [
                "--model_seed=1",
                f"--data_seed={s}",
                f"--wandb_entity={entity}",
                f"--wandb_project={project}",
                f"--wandb_id={run_id}",
            ]
        )

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


if __name__ == "__main__":
    # write the launch script
    partition = "rtx6000"
    script = TEMPLATE.replace("PARTITION_PLACEHOLDER", partition)

    qos = "m5"
    time = QUEUE_TO_TIME[qos]
    script = script.replace("QOS_PLACEHOLDER", qos)
    script = script.replace("TIME_PLACEHOLDER", time)

    # 1) write the launch script varying model initialization
    model_cmds_flat = []
    for sweep_commands in VARY_MODEL_COMMANDS.values():
        model_cmds_flat.extend(iter(sweep_commands.values()))

    model_jobs = [f"{cmd!r}" for cmd in model_cmds_flat]
    model_script = script.replace("JOBS_PLACEHOLDER", "\t" + "\n\t".join(model_jobs))
    model_script = script.replace("ARRAY_PLACEHOLDER", str(len(model_jobs) - 1))

    with open(path.join(REPEATDIR, "launch_vary_model.sh"), "w") as f:
        f.write(model_script)

    # 2) write the launch script varying data initialization
    data_cmds_flat = []
    for sweep_commands in VARY_DATA_COMMANDS.values():
        data_cmds_flat.extend(iter(sweep_commands.values()))

    data_jobs = [f"{cmd!r}" for cmd in data_cmds_flat]
    data_script = script.replace("JOBS_PLACEHOLDER", "\t" + "\n\t".join(data_jobs))
    data_script = script.replace("ARRAY_PLACEHOLDER", str(len(data_jobs) - 1))

    with open(path.join(REPEATDIR, "launch_vary_data.sh"), "w") as f:
        f.write(data_script)
