"""Universal training script for training PINNs.

To see the available command line options of this script, run
```
python train.py --help
```
"""

from argparse import ArgumentParser, Namespace
from functools import partial
from itertools import count
from math import sqrt
from os import makedirs, path
from sys import argv
from time import time
from typing import Iterable, List, Tuple

import wandb
from hessianfree.optimizer import HessianFree
from torch import (
    Tensor,
    cat,
    cuda,
    device,
    dtype,
    float32,
    float64,
    manual_seed,
    rand,
    zeros,
)
from torch.nn import Linear, Module, Sequential, Tanh
from torch.optim import LBFGS

from kfac_pinns_exp import heat_equation, poisson_equation
from kfac_pinns_exp.optim import set_up_optimizer
from kfac_pinns_exp.optim.engd import ENGD
from kfac_pinns_exp.optim.kfac import KFAC
from kfac_pinns_exp.parse_utils import (
    check_all_args_parsed,
    parse_known_args_and_remove_from_argv,
)
from kfac_pinns_exp.poisson_equation import l2_error, square_boundary
from kfac_pinns_exp.train_utils import DataLoader, KillTrigger, LoggingTrigger
from kfac_pinns_exp.utils import latex_float

SUPPORTED_OPTIMIZERS = ["KFAC", "SGD", "Adam", "ENGD", "LBFGS", "HessianFree"]
SUPPORTED_EQUATIONS = ["poisson", "heat"]
SUPPORTED_MODELS = [
    "mlp-tanh-64",
    "mlp-tanh-64-48-32-16",
    "mlp-tanh-64-64-48-48",
    "mlp-tanh-256-256-128-128",
    "mlp-tanh-768-768-512-512",
]
SUPPORTED_BOUNDARY_CONDITIONS = [
    "sin_product",
    "cos_sum",
    "u_weinan",
    "u_weinan_norm",
    "sin_sum",
]

SOLUTIONS = {
    "poisson": {
        "sin_product": poisson_equation.u_sin_product,
        "cos_sum": poisson_equation.u_cos_sum,
        "u_weinan": poisson_equation.u_weinan_prods,
        "u_weinan_norm": poisson_equation.u_weinan_norm,
    },
    "heat": {
        "sin_product": heat_equation.u_sin_product,
        "sin_sum": heat_equation.u_sin_sum,
    },
}
INTERIOR_AND_BOUNDARY_LOSS_EVALUATORS = {
    "poisson": (
        poisson_equation.evaluate_interior_loss,
        poisson_equation.evaluate_boundary_loss,
    ),
    "heat": (
        heat_equation.evaluate_interior_loss,
        heat_equation.evaluate_boundary_loss,
    ),
}


def parse_general_args(verbose: bool = False) -> Namespace:
    """Parse general command-line arguments.

    Args:
        verbose: Whether to print the parsed arguments. Default: `False`.

    Returns:
        A namespace with the parsed arguments.
    """
    DTYPES = {"float32": float32, "float64": float64}
    parser = ArgumentParser(
        description="General training parameters for the Poisson equation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlp-tanh-64",
        choices=SUPPORTED_MODELS,
        help="Which neural network will be used.",
    )
    parser.add_argument(
        "--equation",
        type=str,
        default="poisson",
        choices=SUPPORTED_EQUATIONS,
        help="Which equation and solution will be solved.",
    )
    parser.add_argument(
        "--boundary_condition",
        type=str,
        default="sin_product",
        choices=SUPPORTED_BOUNDARY_CONDITIONS,
        help="Which boundary condition will be used.",
    )
    parser.add_argument(
        "--batch_frequency",
        type=int,
        default=0,
        help="Frequency at which new batches are generated. 0 means never.",
    )
    parser.add_argument(
        "--dim_Omega",
        type=int,
        default=2,
        help="Spatial dimension of the equation's domain Ω.",
    )
    parser.add_argument(
        "--N_Omega",
        type=int,
        default=900,
        help="Number of quadrature points in the domain Ω.",
    )
    parser.add_argument(
        "--N_dOmega",
        type=int,
        default=120,
        help="Number of quadrature points on the boundary ∂Ω.",
    )
    parser.add_argument(
        "--data_seed",
        type=int,
        default=0,
        help="Random seed set before generating the quadrature points.",
    )
    parser.add_argument(
        "--model_seed",
        type=int,
        default=1,
        help="Random seed set before initializing the model's parameters.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=SUPPORTED_OPTIMIZERS,
        help="Which optimizer will be used.",
        required=True,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=DTYPES.keys(),
        default="float64",
        help="Data type for the data and model.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10_000,
        help="Number of training steps.",
    )
    parser.add_argument(
        "--num_seconds",
        type=float,
        default=0.0,
        help="Number of seconds to train. Ignored if `0.0`,"
        + " otherwise disables `num_steps`.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Whether to use Weights & Biases for logging.",
    )
    parser.add_argument(
        "--max_logs",
        type=int,
        default=150,
        help="Maximum number of logs/prints. Ignored if `num_seconds` is non-zero.",
    )
    parser.add_argument(
        "--N_eval",
        type=int,
        default=None,
        help="Number of evaluation points (default: 10 * N_Omega).",
    )
    # plotting-specific arguments
    parser.add_argument(
        "--plot_solution",
        action="store_true",
        help="Whether to plot the learned function and solution during training.",
        default=False,
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="visualize_solution",
        help="Directory to save the plots (only relevant with `--plot_solution`).",
    )
    parser.add_argument(
        "--disable_tex",
        action="store_true",
        default=False,
        help="Disable TeX rendering in plots (only relevant with `--plot_solution`).",
    )
    args = parse_known_args_and_remove_from_argv(parser)

    # overwrite dtype
    args.dtype = DTYPES[args.dtype]

    # set default value for N_eval if not supplied
    if args.N_eval is None:
        args.N_eval = 10 * args.N_Omega

    if verbose:
        print(f"General arguments for the PINN problem: {args}")

    return args


def set_up_layers(model: str, equation: str, dim_Omega: int) -> List[Module]:
    """Set up the layers of the neural network.

    Args:
        model: The name of the model. Must be in `SUPPORTED_MODELS`.
        equation: The name of the equation.
        dim_Omega: The spatial dimension of the domain Ω.

    Returns:
        A list of PyTorch modules representing the layers of the model.

    Raises:
        ValueError: If the model is not supported.
    """
    in_dim = {"poisson": dim_Omega, "heat": dim_Omega + 1}[equation]
    if model == "mlp-tanh-64":
        layers = [
            Linear(in_dim, 64),
            Tanh(),
            Linear(64, 1),
        ]
    elif model == "mlp-tanh-64-48-32-16":
        layers = [
            Linear(in_dim, 64),
            Tanh(),
            Linear(64, 48),
            Tanh(),
            Linear(48, 32),
            Tanh(),
            Linear(32, 16),
            Tanh(),
            Linear(16, 1),
        ]
    elif model == "mlp-tanh-64-64-48-48":
        layers = [
            Linear(in_dim, 64),
            Tanh(),
            Linear(64, 64),
            Tanh(),
            Linear(64, 48),
            Tanh(),
            Linear(48, 48),
            Tanh(),
            Linear(48, 1),
        ]
    elif model == "mlp-tanh-256-256-128-128":
        layers = [
            Linear(in_dim, 256),
            Tanh(),
            Linear(256, 256),
            Tanh(),
            Linear(256, 128),
            Tanh(),
            Linear(128, 128),
            Tanh(),
            Linear(128, 1),
        ]
    elif model == "mlp-tanh-768-768-512-512":
        layers = [
            Linear(in_dim, 768),
            Tanh(),
            Linear(768, 768),
            Tanh(),
            Linear(768, 512),
            Tanh(),
            Linear(512, 512),
            Tanh(),
            Linear(512, 1),
        ]
    else:
        raise ValueError(
            f"Unsupported model: {model}. Supported models: {SUPPORTED_MODELS}"
        )

    return layers


def create_interior_data(
    equation: str, condition: str, dim_Omega: int, num_data: int
) -> Tuple[Tensor, Tensor]:
    """Create random inputs and targets from the PDE's domain.

    Args:
        equation: The name of the equation.
        condition: The name of the boundary/initial condition.
        dim_Omega: The spatial dimension of the PDE's spatial domain Ω.
        num_data: The number of data points to generate.

    Returns:
        A tensor of shape `(num_data, dim)` with random input data where `dim` is the
        dimensionality of the PDE's entire domain (i.e. might add a time axis) and a
        tensor of shape `(num_data, 1)` containing the targets.

    Raises:
        NotImplementedError: If the combination of equation and condition is not
            supported.
    """
    dim = {"poisson": dim_Omega, "heat": dim_Omega + 1}[equation]
    X = rand(num_data, dim)
    if equation == "poisson" and condition in {
        "sin_product",
        "cos_sum",
        "u_weinan",
        "u_weinan_norm",
    }:
        f = {
            "sin_product": poisson_equation.f_sin_product,
            "cos_sum": poisson_equation.f_cos_sum,
            "u_weinan": poisson_equation.f_weinan_prods,
            "u_weinan_norm": poisson_equation.f_weinan_norm,
        }[condition]
        y = f(X)
    elif equation == "heat" and condition in {
        "sin_product",
        "sin_sum",
    }:
        y = zeros(num_data, 1)
    else:
        raise NotImplementedError(
            f"Equation {equation} with condition {condition} is not supported."
        )

    return X, y


def create_condition_data(
    equation: str, condition: str, dim_Omega: int, num_data: int
) -> Tuple[Tensor, Tensor]:
    """Create data points to enforce conditions on a PDE.

    Conditions can be boundary conditions or initial value conditions.

    Args:
        equation: The name of the equation.
        condition: The name of the boundary/initial condition.
        dim_Omega: The spatial dimension of the PDE's spatial domain Ω.
        num_data: The number of data points to generate.

    Returns:
        A tuple `(X, y)` with the input data and labels for the boundary/initial
        condition.

    Raises:
        NotImplementedError: If the combination of equation and condition is not
            supported.
    """
    if equation == "poisson" and condition in {
        "sin_product",
        "cos_sum",
        "u_weinan",
        "u_weinan_norm",
    }:
        # boundary condition
        X_dOmega = square_boundary(num_data, dim_Omega)
    elif equation == "heat" and condition in {
        "sin_product",
        "sin_sum",
    }:
        # boundary condition
        X_dOmega1 = heat_equation.square_boundary_random_time(num_data // 2, dim_Omega)
        # initial value condition
        X_dOmega2 = heat_equation.unit_square_at_start(num_data // 2, dim_Omega)
        X_dOmega = cat([X_dOmega1, X_dOmega2])
    else:
        raise NotImplementedError(
            f"Equation {equation} and condition {condition} not supported."
        )

    u = SOLUTIONS[equation][condition]
    return X_dOmega, u(X_dOmega)


def create_data_loader(
    frequency: int,
    loss_type: str,
    equation: str,
    condition: str,
    dim_Omega: int,
    num_data: int,
    dev: device,
    dt: dtype,
) -> Iterable[Tuple[Tensor, Tensor]]:
    """Create a data loader for one of the losses.

    Args:
        frequency: How many steps before the data loader samples a new batch.
        loss_type: For which type of loss to generate data. Can be either `'interior'`
            or `'condition'`.
        equation: The name of the equation.
        condition: The name of the boundary/initial condition.
        dim_Omega: The spatial dimension of the PDE's spatial domain Ω.
        num_data: The number of data points to generate.
        dev: The device on which the tensors returned by the data loader live on.
        dt: The data type of the tensors returned by the data loader.

    Returns:
        A data loader for the specified loss which generates batched `(X, y)` pairs.
    """
    data_func = {"interior": create_interior_data, "condition": create_condition_data}[
        loss_type
    ]
    data_func = partial(data_func, equation, condition, dim_Omega, num_data)
    return DataLoader(data_func, dev, dt, frequency)


def main():  # noqa: C901
    """Execute training with the specified command line arguments."""
    # NOTE Do not move this down as the parsers remove arguments from argv
    cmd = " ".join(["python"] + argv)
    args = parse_general_args(verbose=True)
    dev, dt = device("cuda" if cuda.is_available() else "cpu"), args.dtype
    print(f"Running on device {str(dev)} in dtype {dt}.")
    if args.plot_solution:
        print(f"Saving visualizations of the solution in {args.plot_dir}.")

    # DATA LOADERS
    manual_seed(args.data_seed)
    equation, condition = args.equation, args.boundary_condition
    dim_Omega, N_Omega, N_dOmega = args.dim_Omega, args.N_Omega, args.N_dOmega

    # for satisfying the PDE on the domain
    interior_train_data_loader = iter(
        create_data_loader(
            args.batch_frequency,
            "interior",
            equation,
            condition,
            dim_Omega,
            N_Omega,
            dev,
            dt,
        )
    )
    interior_eval_data_loader = iter(
        create_data_loader(
            0,  # fixed evaluation data
            "interior",
            equation,
            condition,
            dim_Omega,
            args.N_eval,
            dev,
            dt,
        )
    )
    # for satisfying boundary and (maybe) initial conditions
    condition_train_data_loader = iter(
        create_data_loader(
            args.batch_frequency,
            "condition",
            equation,
            condition,
            dim_Omega,
            N_dOmega,
            dev,
            dt,
        )
    )

    # NEURAL NET
    manual_seed(args.model_seed)
    layers = set_up_layers(args.model, equation, dim_Omega)
    layers = [layer.to(dev, dt) for layer in layers]
    model = Sequential(*layers).to(dev)
    print(f"Model: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # OPTIMIZER
    optimizer, optimizer_args = set_up_optimizer(
        layers, args.optimizer, equation, verbose=True
    )
    check_all_args_parsed()

    # check that the equation was correctly passed to PDE-aware optimizers
    if isinstance(optimizer, (KFAC, ENGD)):
        assert optimizer.equation == equation

    if args.wandb:
        config = vars(args) | vars(optimizer_args) | {"cmd": cmd}
        wandb.init(config=config)

    # functions used to evaluate the interior and boundary/condition losses
    eval_interior_loss, eval_boundary_loss = INTERIOR_AND_BOUNDARY_LOSS_EVALUATORS[
        equation
    ]

    # TRAINING
    logging_trigger = LoggingTrigger(args.num_steps, args.max_logs, args.num_seconds)
    kill_trigger = KillTrigger(args.num_steps, args.num_seconds)
    start = time()

    for step in count():
        # load next batch of data
        X_Omega, y_Omega = next(interior_train_data_loader)
        X_dOmega, y_dOmega = next(condition_train_data_loader)

        optimizer.zero_grad()

        if isinstance(optimizer, (KFAC, ENGD)):
            loss_interior, loss_boundary = optimizer.step(
                X_Omega, y_Omega, X_dOmega, y_dOmega
            )
        elif isinstance(optimizer, LBFGS):
            # LBFGS requires a closure

            def closure() -> Tensor:
                """Evaluate the loss on the current data and model parameters.

                Note:
                    It is okay to ignore the flake8 warning B023 that this function
                    will change if we change the loop variables
                    `X_Omega, y_Omega, X_dOmega, y_dOmega` as we only use the closure
                    within one iteration of the loop.

                Returns:
                    The loss.
                """
                optimizer.zero_grad()
                # compute the interior loss' gradient
                loss_interior, _, _ = eval_interior_loss(
                    layers,
                    X_Omega,  # noqa: B023, see note above
                    y_Omega,  # noqa: B023, see note above
                )
                loss_interior.backward()

                # compute the boundary loss' gradient
                loss_boundary, _, _ = eval_boundary_loss(
                    layers,
                    X_dOmega,  # noqa: B023, see note above
                    y_dOmega,  # noqa: B023, see note above
                )
                loss_boundary.backward()

                # HOTFIX Append the interior and boundary loss as arguments
                # so we can extract them for logging and plotting
                loss = loss_interior + loss_boundary
                loss._loss_interior = loss_interior
                loss._loss_boundary = loss_boundary

                return loss

            loss_original = optimizer.step(closure=closure)
            loss_interior = loss_original._loss_interior
            loss_boundary = loss_original._loss_boundary

        elif isinstance(optimizer, HessianFree):
            # HessianFree requires a closure that produces the linearization
            # point and the loss

            # store the loss values of the closure because we want to log them
            # at the current position.
            loss_storage = []

            def forward(
                loss_storage: List[Tuple[Tensor, Tensor]]
            ) -> Tuple[Tensor, Tensor]:
                """Compute the linearization point for the GGN and the loss.

                Args:
                    loss_storage: A list to append the the interior and boundary loss.

                Note:
                    It is okay to ignore the flake8 warning B023 that this function
                    will change if we change the loop variables
                    `X_Omega, y_Omega, X_dOmega, y_dOmega` as we only use the closure
                    within one iteration of the loop.

                Returns:
                    The linearization point and the loss.
                """
                loss_interior, residual_interior, _ = eval_interior_loss(
                    layers,
                    X_Omega,  # noqa: B023, see note above
                    y_Omega,  # noqa: B023, see note above
                )
                loss_boundary, residual_boundary, _ = eval_boundary_loss(
                    layers,
                    X_dOmega,  # noqa: B023, see note above
                    y_dOmega,  # noqa: B023, see note above
                )
                # we want to linearize residual w.r.t. the parameters to obtain
                # the GGN. This established the connection between the loss and
                # the concatenated boundary and interior residuals.
                residual = cat(
                    [
                        residual_interior / sqrt(residual_interior.numel()),
                        residual_boundary / sqrt(residual_boundary.numel()),
                    ]
                )
                loss = 0.5 * (residual**2).sum()

                # HOTFIX Append the interior and boundary loss to loss_storage
                # so we can extract them for logging and plotting
                loss_storage.append((loss_interior.detach(), loss_boundary.detach()))

                return loss, residual

            forward = partial(forward, loss_storage=loss_storage)
            optimizer.step(forward)
            loss_interior, loss_boundary = loss_storage[0]

        else:
            # compute the interior loss' gradient
            loss_interior, _, _ = eval_interior_loss(layers, X_Omega, y_Omega)
            loss_interior.backward()
            # compute the boundary loss' gradient
            loss_boundary, _, _ = eval_boundary_loss(layers, X_dOmega, y_dOmega)
            loss_boundary.backward()
            optimizer.step()

        now = time()
        elapsed = now - start
        loss_boundary, loss_interior = loss_boundary.item(), loss_interior.item()
        loss = loss_interior + loss_boundary

        if logging_trigger.should_log(step) or kill_trigger.should_kill(step, elapsed):
            # load next batch of evaluation data
            X_Omega_eval, _ = next(interior_eval_data_loader)
            # function to evaluate the known solution
            u = SOLUTIONS[equation][condition]
            l2 = l2_error(model, X_Omega_eval, u)
            print(
                f"Step: {step:07g},"
                + f" Loss: {loss},"
                + f" L2 Error: {l2},"
                + f" Interior: {loss_interior},"
                + f" Boundary: {loss_boundary},"
                + f" Time: {elapsed:.1f}s",
                flush=True,
            )
            if args.wandb:
                wandb.log(
                    {
                        "step": step,
                        "loss": loss,
                        "loss_interior": loss_interior,
                        "loss_boundary": loss_boundary,
                        "l2_error": l2,
                        "time": elapsed,
                    }
                )
            if args.plot_solution:
                fig_path = path.join(
                    args.plot_dir,
                    f"{equation}_{dim_Omega}d_{condition}_{args.model}"
                    + f"_{args.optimizer}_step{step:07g}.pdf",
                )
                fig_title = (
                    f"Step: ${step}$, Loss: ${latex_float(loss)}$,"
                    + f" $L_2$ loss: ${latex_float(l2.item())}$"
                )
                makedirs(args.plot_dir, exist_ok=True)
                plot_fn = {
                    "poisson": poisson_equation.plot_solution,
                    "heat": heat_equation.plot_solution,
                }[equation]
                plot_fn(
                    condition,
                    dim_Omega,
                    model,
                    fig_path,
                    title=fig_title,
                    usetex=not args.disable_tex,
                )

        if kill_trigger.should_kill(step, elapsed):
            return


if __name__ == "__main__":
    main()
