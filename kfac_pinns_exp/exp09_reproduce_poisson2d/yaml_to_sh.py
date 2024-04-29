"""Create a shell script from the sweep .yaml configuration."""

from argparse import ArgumentParser

from kfac_pinns_exp.utils import run_verbose

QUEUE_TO_TIME = {
    "normal": "16:00:00",
    "m": "12:00:00",
    "m2": "08:00:00",
    "m3": "04:00:00",
    "m4": "02:00:00",
    "m5": "01:00:00",
}

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Create shell script from sweep .yaml configuration."
    )
    parser.add_argument("yaml_file", type=str, help="Path to the .yaml file.")
    parser.add_argument("sweep_name", type=str, help="Name of the sweep.")
    parser.add_argument(
        "--qos",
        type=str,
        choices=QUEUE_TO_TIME.keys(),
        default="m4",
        help="Slurm QOS for the job.",
    )
    args = parser.parse_args()

    cmd = [
        "wandb",
        "sweep",
        f"--name={args.sweep_name}",
        args.yaml_file,
        "--entity=kfac-pinns",
    ]

    # run the wandb command
    job = run_verbose(cmd)

    lines = job.stderr.split("\n")
    trigger = "wandb: Run sweep agent with: wandb agent "
    (line,) = [line for line in lines if line.startswith(trigger)]
    line = line.replace(trigger, "")
    line = f"wandb agent --count 1 {line}"

    # create the .sh file
    TEMPLATE = f"""#!/bin/bash
#SBATCH --partition=rtx6000,t4v1,t4v2
#SBATCH --qos={args.qos}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time={QUEUE_TO_TIME[args.qos]}
#SBATCH --mem-per-cpu=8G
#SBATCH --array=1-64%16

source  ~/anaconda3/etc/profile.d/conda.sh
conda activate kfac_pinns_exp

{line}"""

    sh_file = args.yaml_file.replace(".yaml", ".sh")
    with open(sh_file, "w") as f:
        f.write(TEMPLATE)
