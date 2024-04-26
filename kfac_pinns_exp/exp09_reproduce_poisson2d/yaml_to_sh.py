"""Create a shell script from the sweep .yaml configuration."""

from argparse import ArgumentParser

from kfac_pinns_exp.utils import run_verbose

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Create shell script from sweep .yaml configuration."
    )
    parser.add_argument("yaml_file", type=str, help="Path to the .yaml file.")
    parser.add_argument("sweep_name", type=str, help="Name of the sweep.")

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

    # get the command from stderr
    trigger = "wandb: Run sweep agent with: wandb agent "
    lines = job.stderr.split("\n")
    (line,) = [line for line in lines if line.startswith(trigger)]
    line = line.replace(trigger, "")
    line = f"wandb agent --count 1 {line}"

    # create the .sh file
    TEMPLATE = (
        r"""#!/bin/bash
#SBATCH --partition=rtx6000,t4v1,t4v2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=1-64%5

source  ~/anaconda3/etc/profile.d/conda.sh
conda activate kfac_pinns_exp

"""
        + line
    )

    sh_file = args.yaml_file.replace(".yaml", ".sh")
    with open(sh_file, "w") as f:
        f.write(TEMPLATE)
