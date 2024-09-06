"""Test visualization of solutions from checkpoints."""

from typing import List

from pytest import mark

from kfac_pinns_exp import plot_solution, train
from kfac_pinns_exp.utils import run_verbose

ARGS = [
    # train and checkpoint for each logged step
    *[
        [
            "--num_steps=3",
            f"--dim_Omega={dim_Omega}",
            f"--equation={equation}",
            f"--boundary_condition={condition}",
            "--optimizer=SGD",
            "--SGD_lr=0.1",
            "--save_checkpoints",
        ]
        for dim_Omega, equation, condition in [
            (1, "poisson", "sin_product"),
            (2, "poisson", "sin_product"),
            (1, "poisson", "cos_sum"),
            (2, "poisson", "cos_sum"),
            (1, "heat", "sin_product"),
            (1, "heat", "sin_sum"),
            (1, "fokker-planck-isotropic", "gaussian"),
            (1, "log-fokker-planck-isotropic", "gaussian"),
        ]
    ],
]
ARG_IDS = ["_".join(cmd) for cmd in ARGS]


@mark.parametrize("arg", ARGS, ids=ARG_IDS)
def test_plot_solution(arg: List[str]):
    """Train and save checkpoints, then visualize the solution.

    Args:
        arg: The command-line arguments to pass to the script.
    """
    run_verbose(["python", train.__file__] + arg)

    plot_args = [
        "--checkpoint_dir=checkpoints",
        "--disable_tex",  # for Github actions (no LaTeX available)
    ]
    run_verbose(["python", plot_solution.__file__] + plot_args)
