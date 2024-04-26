"""Test the training script (integration test)."""

from itertools import product
from subprocess import CalledProcessError, run
from typing import List

from pytest import mark

from kfac_pinns_exp import train


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


ARGS = [
    # train with ENGD and on different equations
    *[
        [
            "--num_steps=3",
            "--optimizer=ENGD",
            f"--equation={equation}",
            "--ENGD_ema_factor=0.99",
            "--ENGD_damping=0.0001",
            "--ENGD_lr=0.1",
            f"--ENGD_approximation={approximation}",
        ]
        for equation, approximation in product(
            ["poisson", "heat"], ["full", "per_layer", "diagonal"]
        )
    ],
    # train with KFAC
    *[
        [
            "--num_steps=10",
            "--optimizer=KFAC",
            f"--equation={equation}",
            "--KFAC_T_kfac=2",
            "--KFAC_T_inv=4",
            "--KFAC_ema_factor=0.95",
            "--KFAC_damping=0.01",
            "--KFAC_lr=0.1",
            f"--KFAC_ggn_type={ggn_type}",
        ]
        for equation, ggn_type in product(
            ["poisson", "heat"], ["type-2", "empirical", "forward-only"]
        )
    ],
    # train with SGD
    *[
        [
            "--num_steps=3",
            "--optimizer=SGD",
            f"--equation={equation}",
            "--SGD_lr=0.1",
            "--SGD_momentum=0.9",
        ]
        for equation in ["poisson", "heat"]
    ],
    # train with Adam
    [
        "--num_steps=3",
        "--optimizer=Adam",
        "--Adam_lr=0.01",
        "--Adam_beta1=0.8",
        "--Adam_beta2=0.99",
    ],
    # train with LBFGS
    *[
        [
            "--num_steps=3",
            "--optimizer=LBFGS",
            f"--equation={equation}",
        ]
        for equation in ["poisson", "heat"]
    ],
    # train with HessianFree
    *[
        [
            "--num_steps=3",
            "--optimizer=HessianFree",
            f"--equation={equation}",
        ]
        for equation in ["poisson", "heat"]
    ],
    # train with a deeper net
    [
        "--num_steps=3",
        "--optimizer=SGD",
        "--SGD_lr=0.1",
        "--model=mlp-tanh-64-48-32-16",
    ],
    # train with different boundary conditions
    [
        "--num_steps=3",
        "--optimizer=SGD",
        "--SGD_lr=0.1",
        "--boundary_condition=cos_sum",
    ],
    # train and visualize the solutions for each logged step
    *[
        [
            "--num_steps=3",
            f"--dim_Omega={dim_Omega}",
            f"--equation={equation}",
            f"--boundary_condition={condition}",
            "--optimizer=SGD",
            "--SGD_lr=0.1",
            "--plot_solution",
            "--disable_tex",  # for Github actions (no LaTeX available)
        ]
        for dim_Omega, equation, condition in [
            (1, "poisson", "sin_product"),
            (2, "poisson", "sin_product"),
            (1, "poisson", "cos_sum"),
            (2, "poisson", "cos_sum"),
        ]
    ],
]
ARG_IDS = ["_".join(cmd) for cmd in ARGS]


@mark.parametrize("arg", ARGS, ids=ARG_IDS)
def test_train(arg: List[str]):
    """Execute the training script.

    Args:
        arg: The command-line arguments to pass to the script.
    """
    _run(["python", train.__file__] + arg)
