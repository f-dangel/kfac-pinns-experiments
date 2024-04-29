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


def create_sbatch_script(
    savepath: str,
    cmd: str,
    qos: str = "m4",
    array: int = 64,
    array_max_active: int = 16,
    partition: str = "rtx6000",
):
    """Create an sbatch script containing the given command.

    Args:
        savepath: Path to save the sbatch script.
        cmd: Command to run in the sbatch script.
        qos: Slurm QOS for the job. Default is `"m4"`.
        array: Size of the job array. Default: `64`.
        array_max_active: Maximum number of active tasks in the array. Default: `16`.
        partition: Slurm partition for the job. Default: `"rtx6000"`.
    """
    script = f"""#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time={QUEUE_TO_TIME[qos]}
#SBATCH --mem-per-cpu=8G
#SBATCH --array=1-{array}%{min(array, array_max_active)}

source  ~/anaconda3/etc/profile.d/conda.sh
conda activate kfac_pinns_exp

{cmd}"""

    with open(savepath, "w") as f:
        f.write(script)


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

    sh_file = args.yaml_file.replace(".yaml", ".sh")
    create_sbatch_script(sh_file, line, args.qos)
