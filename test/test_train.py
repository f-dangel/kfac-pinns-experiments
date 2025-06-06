"""Test the training script (integration test)."""

from itertools import product
from typing import List

from pytest import mark

from kfac_pinns_exp import train
from kfac_pinns_exp.utils import run_verbose

ARGS = [
    # train with ENGD and on different equations
    *[
        [
            "--num_steps=3",
            "--optimizer=ENGD",
            f"--equation={equation}",
            f"--boundary_condition={condition}",
            "--ENGD_ema_factor=0.99",
            "--ENGD_damping=0.0001",
            "--ENGD_lr=0.1",
            f"--ENGD_approximation={approximation}",
        ]
        for (equation, condition), approximation in product(
            [
                ("poisson", "sin_product"),
                ("heat", "sin_product"),
                ("fokker-planck-isotropic", "gaussian"),
                ("log-fokker-planck-isotropic", "gaussian"),
            ],
            ["full", "per_layer", "diagonal"],
        )
    ],
    # train with KFAC
    *[
        [
            "--num_steps=10",
            "--optimizer=KFAC",
            f"--equation={equation}",
            f"--boundary_condition={condition}",
            "--KFAC_T_kfac=2",
            "--KFAC_T_inv=4",
            "--KFAC_ema_factor=0.95",
            "--KFAC_damping=0.01",
            "--KFAC_lr=0.1",
            f"--KFAC_ggn_type={ggn_type}",
        ]
        for (equation, condition), ggn_type in product(
            [
                ("poisson", "sin_product"),
                ("heat", "sin_product"),
                ("fokker-planck-isotropic", "gaussian"),
                ("log-fokker-planck-isotropic", "gaussian"),
            ],
            ["type-2", "empirical", "forward-only"],
        )
    ],
    # train with SGD
    *[
        [
            "--num_steps=3",
            "--optimizer=SGD",
            f"--equation={equation}",
            f"--boundary_condition={condition}",
            "--SGD_lr=0.1",
            "--SGD_momentum=0.9",
        ]
        for equation, condition in [
            ("poisson", "sin_product"),
            ("heat", "sin_product"),
            ("fokker-planck-isotropic", "gaussian"),
            ("log-fokker-planck-isotropic", "gaussian"),
        ]
    ],
    # train with Adam
    *[
        [
            "--num_steps=3",
            "--optimizer=Adam",
            "--Adam_lr=0.01",
            "--Adam_beta1=0.8",
            "--Adam_beta2=0.99",
            f"--equation={equation}",
            f"--boundary_condition={condition}",
        ]
        for equation, condition in [
            ("poisson", "sin_product"),
            ("heat", "sin_product"),
            ("fokker-planck-isotropic", "gaussian"),
            ("log-fokker-planck-isotropic", "gaussian"),
        ]
    ],
    # train with LBFGS
    *[
        [
            "--num_steps=3",
            "--optimizer=LBFGS",
            f"--equation={equation}",
            f"--boundary_condition={condition}",
        ]
        for (equation, condition) in [
            ("poisson", "sin_product"),
            ("heat", "sin_product"),
            ("fokker-planck-isotropic", "gaussian"),
            ("log-fokker-planck-isotropic", "gaussian"),
        ]
    ],
    # train with HessianFree
    *[
        [
            "--num_steps=3",
            "--optimizer=HessianFree",
            f"--equation={equation}",
            f"--boundary_condition={condition}",
        ]
        for (equation, condition) in [
            ("poisson", "sin_product"),
            ("heat", "sin_product"),
            ("fokker-planck-isotropic", "gaussian"),
            ("log-fokker-planck-isotropic", "gaussian"),
        ]
    ],
    # train with HessianFreeCached (w/o KFAC pre-conditioning)
    *[
        [
            "--num_steps=3",
            "--optimizer=HessianFreeCached",
            f"--equation={equation}",
            f"--boundary_condition={condition}",
        ]
        + preconditioning
        for (equation, condition), preconditioning in product(
            [
                ("poisson", "sin_product"),
                ("heat", "sin_product"),
                ("fokker-planck-isotropic", "gaussian"),
                ("log-fokker-planck-isotropic", "gaussian"),
            ],
            [["--HessianFreeCached_kfac_preconditioner"], []],
        )
    ],
    # train with SPRING
    *[
        [
            "--num_steps=3",
            "--optimizer=SPRING",
            "--SPRING_lr=5e-1",
            "--SPRING_damping=1e-5",
            "--SPRING_decay_factor=0.9",
            "--SPRING_norm_constraint=1",
            f"--equation={equation}",
            f"--boundary_condition={condition}",
        ]
        for (equation, condition) in [
            ("poisson", "sin_product"),
            ("heat", "sin_product"),
            ("fokker-planck-isotropic", "gaussian"),
            ("log-fokker-planck-isotropic", "gaussian"),
        ]
    ],
    # train with a deeper net
    *[
        [
            "--num_steps=3",
            "--optimizer=SGD",
            "--SGD_lr=0.1",
            f"--model={model}",
        ]
        for model in [
            "mlp-tanh-64-48-32-16",
            "mlp-tanh-64-64-48-48",
            "mlp-tanh-256-256-128-128",
            "mlp-tanh-768-768-512-512",
        ]
    ],
    # train with different boundary conditions
    [
        "--num_steps=3",
        "--optimizer=SGD",
        "--SGD_lr=0.1",
        "--boundary_condition=cos_sum",
    ],
    # train and store checkpoints
    [
        "--num_steps=3",
        "--optimizer=SGD",
        "--SGD_lr=0.1",
        "--save_checkpoints",
    ],
    [
        "--num_steps=3",
        "--optimizer=Adam",
        "--Adam_lr=0.1",
        "--save_checkpoints",
        "--checkpoint_steps",
        "0",
        "1",
    ],
    # train with KFAC+momentum
    [
        "--num_steps=3",
        "--optimizer=KFAC",
        "--KFAC_damping=0.01",
        "--KFAC_momentum=0.1",
    ],
    # train with KFAC+automatic learning rate and momentum
    [
        "--num_steps=3",
        "--optimizer=KFAC",
        "--KFAC_damping=0.01",
        "--KFAC_lr=auto",
    ],
    # train with KFAC+trace-norm damping heuristic
    [
        "--num_steps=3",
        "--optimizer=KFAC",
        "--KFAC_damping=0.01",
        "--KFAC_damping_heuristic=trace-norm",
    ],
    # train with SGD + new batches every 2 steps
    [
        "--num_steps=5",
        "--optimizer=SGD",
        "--SGD_lr=0.1",
        "--SGD_momentum=0.9",
        "--batch_frequency=2",
    ],
]
ARG_IDS = ["_".join(cmd) for cmd in ARGS]


@mark.parametrize("arg", ARGS, ids=ARG_IDS)
def test_train(arg: List[str]):
    """Execute the training script.

    Args:
        arg: The command-line arguments to pass to the script.
    """
    run_verbose(["python", train.__file__] + arg)
