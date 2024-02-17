"""Execute the training script in `exp09` (integration test)."""

from subprocess import CalledProcessError, run
from typing import List

from kfac_pinns_exp.exp09_kfac_optimizer import train


def _run(cmd: List[str]):
    """Run the command and print the output/stderr if it fails.

    Args:
        cmd: The command to run.

    Raises:
        CalledProcessError: If the command fails.
    """
    try:
        run(cmd, capture_output=True, text=True, check=True)
    except CalledProcessError as e:
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        raise e


def test_run_exp09_train_with_KFACForPINNs():
    """Execute the training script. Train with KFACForPINNs."""
    cmd = [
        "python",
        train.__file__,
        "--num_steps=50",
        "--optimizer=KFACForPINNs",
        "--T_kfac=2",
        "--T_inv=6",
        "--ema_factor=0.95",
        "--damping=0.01",
        "--lr=0.1",
    ]
    _run(cmd)


def test_run_exp09_train_with_SGD():
    """Execute the training script. Train with SGD."""
    cmd = [
        "python",
        train.__file__,
        "--num_steps=50",
        "--optimizer=SGD",
        "--lr=0.1",
        "--momentum=0.9",
    ]
    _run(cmd)


def test_run_exp09_train_with_Adam():
    """Execute the training script. Train with Adam."""
    cmd = [
        "python",
        train.__file__,
        "--num_steps=50",
        "--optimizer=Adam",
        "--lr=0.01",
        "--beta1=0.8",
        "--beta2=0.99",
    ]
    _run(cmd)
