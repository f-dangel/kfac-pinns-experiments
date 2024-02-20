"""Train with KFAC for the Poisson equation."""

from argparse import ArgumentParser, Namespace
from math import log10, pi
from sys import argv
from time import time
from typing import List, Tuple

import wandb
from torch import (
    Tensor,
    cos,
    cuda,
    device,
    float32,
    float64,
    logspace,
    manual_seed,
    prod,
    rand,
    randint,
    sin,
    zeros_like,
)
from torch.nn import Linear, Module, Sequential, Tanh
from torch.optim import SGD, Adam, Optimizer

from kfac_pinns_exp.exp09_kfac_optimizer.engd import ENGD, parse_ENGD_args
from kfac_pinns_exp.exp09_kfac_optimizer.kfac import KFAC, parse_KFAC_args
from kfac_pinns_exp.exp09_kfac_optimizer.utils import (
    parse_Adam_args,
    parse_known_args_and_remove_from_argv,
    parse_SGD_args,
)
from kfac_pinns_exp.poisson_equation import (
    evaluate_boundary_loss,
    evaluate_interior_loss,
)

SUPPORTED_OPTIMIZERS = ["KFAC", "SGD", "Adam", "ENGD"]


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
        "--dim_Omega",
        type=int,
        default=2,
        help="Spatial dimension of the Poisson equation's domain Ω.",
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
        "--wandb",
        action="store_true",
        help="Whether to use Weights & Biases for logging.",
    )
    parser.add_argument(
        "--max_logs",
        type=int,
        default=150,
        help="Maximum number of logs/prints.",
    )
    args = parse_known_args_and_remove_from_argv(parser)

    # overwrite dtype
    args.dtype = DTYPES[args.dtype]

    if verbose:
        print(f"General arguments for the Poisson equation: {args}")

    return args


def check_all_args_parsed():
    """Make sure all command line arguments were parsed.

    Raises:
        ValueError: If there are unparsed arguments.
    """
    if len(argv) != 1:
        raise ValueError(f"The following arguments could not be parsed: {argv[1:]}.")


def set_up_optimizer(
    layers: List[Module], optimizer: str, verbose: bool = False
) -> Tuple[Optimizer, Namespace]:
    """Parse arguments for the specified optimizer and construct it.

    Args:
        layers: The layers of the model.
        optimizer: The name of the optimizer to be used.
        verbose: Whether to print the parsed arguments. Default: `False`.

    Returns:
        The optimizer and the parsed arguments.

    Raises:
        NotImplementedError: If the optimizer is not supported.
    """
    if optimizer == "KFAC":
        cls, args = KFAC, parse_KFAC_args(verbose=verbose)
        args_dict = vars(args)  # each key has 'KFAC_' as prefix
        args_dict = {
            key.removeprefix("KFAC_"): value for key, value in args_dict.items()
        }
        return cls(layers, **args_dict), args

    elif optimizer == "ENGD":
        cls, args = ENGD, parse_ENGD_args(verbose=verbose)
        args_dict = vars(args)  # each key has 'ENGD_' as prefix
        args_dict = {
            key.removeprefix("ENGD_"): value for key, value in args_dict.items()
        }
        return cls(Sequential(*layers), **args_dict), args

    else:
        if optimizer == "Adam":
            cls, args = Adam, parse_Adam_args(verbose=verbose)
        elif optimizer == "SGD":
            cls, args = SGD, parse_SGD_args(verbose=verbose)
        else:
            raise NotImplementedError(f"Unsupported optimizer: {optimizer}.")
        params = sum((list(layer.parameters()) for layer in layers), [])
        return cls(params, **vars(args)), args


# utilities for the Poisson equation


# TODO Use code from exp02 once it is merged
def square_boundary(N: int, dim: int) -> Tensor:
    """Returns quadrature points on the boundary of a square.

    Args:
        N: Number of quadrature points.
        dim: Dimension of the Square.

    Returns:
        A tensor of shape (N, dim) that consists of uniformly drawn
        quadrature points.
    """
    X = rand(N, dim)

    dimensions = randint(0, dim, (N,))
    sides = randint(0, 2, (N,))

    for i in range(N):
        X[i, dimensions[i]] = sides[i].float()

    return X


# Right-hand side of the Poisson equation
# TODO Use code from exp02 once it is merged
def f(X: Tensor) -> Tensor:
    """The right-hand side of the Poisson equation we aim to solve.

    Args:
        X: Batched quadrature points of shape (N, d_Omega).

    Returns:
        The function values as tensor of shape (N, 1).
    """
    d = X.shape[1:].numel()

    return d * pi**2 * prod(sin(pi * X), dim=1, keepdim=True)


# Manufactured solution
# TODO Use code from exp02 once it is merged
def u(X: Tensor) -> Tensor:
    """The solution of the Poisson equation we aim to solve.

    Args:
        X: Batched quadrature points of shape (N, d_Omega).

    Returns:
        The function values as tensor of shape (N, 1).
    """
    return prod(cos(pi * X), dim=1, keepdim=True)


def main():
    args = parse_general_args(verbose=True)

    dev = device("cuda" if cuda.is_available() else "cpu")
    dt = args.dtype
    print(f"Running on device {str(dev)}")

    manual_seed(args.data_seed)
    # quadrature points = data
    # interior
    X_Omega = rand(args.N_Omega, args.dim_Omega).to(dev, dt)
    y_Omega = -f(X_Omega)
    # boundary
    X_dOmega = square_boundary(args.N_dOmega, args.dim_Omega).to(dev, dt)
    y_dOmega = u(X_dOmega)
    y_dOmega = zeros_like(y_dOmega)

    # neural net
    manual_seed(args.model_seed)
    layers = [
        Linear(args.dim_Omega, 64),
        Tanh(),
        Linear(64, 1),
    ]
    layers = [layer.to(dev, dt) for layer in layers]
    model = Sequential(*layers).to(dev)
    print(f"Model: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # optimizer and its hyper-parameters
    optimizer, optimizer_args = set_up_optimizer(layers, args.optimizer, verbose=True)
    check_all_args_parsed()

    if args.wandb:
        config = vars(args) | vars(optimizer_args)
        wandb.init(
            entity="kfac-pinns",
            project="exp09_kfac_optimizer",
            config=config,
        )

    logged_steps = {
        int(s) for s in logspace(0, log10(args.num_steps - 1), args.max_logs - 1).int()
    } | {0}

    start = time()

    # training loop
    for step in range(args.num_steps):
        optimizer.zero_grad()

        if isinstance(optimizer, (KFAC, ENGD)):
            loss_interior, loss_boundary = optimizer.step(
                X_Omega, y_Omega, X_dOmega, y_dOmega
            )

        else:
            # compute the interior loss' gradient
            loss_interior, _, _ = evaluate_interior_loss(layers, X_Omega, y_Omega)
            loss_interior.backward()
            # compute the boundary loss' gradient
            loss_boundary, _, _ = evaluate_boundary_loss(layers, X_dOmega, y_dOmega)
            loss_boundary.backward()
            optimizer.step()

        now = time()
        expired = now - start
        loss_boundary, loss_interior = loss_boundary.item(), loss_interior.item()
        loss = loss_interior + loss_boundary

        if step in logged_steps:
            print(
                f"Step: {step:06g}/{args.num_steps:06g},"
                + f" Loss: {loss:.8f},"
                + f" Interior: {loss_interior:.8f},"
                + f" Boundary: {loss_boundary:.8f},"
                + f" Time: {expired:.1f}s",
                flush=True,
            )
            if args.wandb:
                wandb.log(
                    {
                        "step": step,
                        "loss": loss,
                        "loss_interior": loss_interior,
                        "loss_boundary": loss_boundary,
                        "time": expired,
                    }
                )


if __name__ == "__main__":
    main()
