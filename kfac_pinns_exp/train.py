"""Universal training script for training PINNs.

To see the available command line options of this script, run
```
python train.py --help
```
"""

from argparse import ArgumentParser, Namespace
from functools import partial
from math import log10
from time import time
from typing import List, Tuple

import wandb
from hessianfree.optimizer import HessianFree
from torch import (
    Tensor,
    cat,
    cuda,
    device,
    float32,
    float64,
    logspace,
    manual_seed,
    rand,
)
from torch.nn import Linear, Module, Sequential, Tanh
from torch.optim import LBFGS

from kfac_pinns_exp import poisson_equation
from kfac_pinns_exp.optim import set_up_optimizer
from kfac_pinns_exp.optim.engd import ENGD
from kfac_pinns_exp.optim.kfac import KFAC
from kfac_pinns_exp.parse_utils import (
    check_all_args_parsed,
    parse_known_args_and_remove_from_argv,
)
from kfac_pinns_exp.poisson_equation import (
    evaluate_boundary_loss,
    evaluate_interior_loss,
    l2_error,
)

SUPPORTED_OPTIMIZERS = ["KFAC", "SGD", "Adam", "ENGD", "LBFGS", "HessianFree"]
SUPPORTED_EQUATIONS = ["poisson", "poisson_cos_sum"]
SUPPORTED_MODELS = ["mlp-tanh-64", "mlp-tanh-64-48-32-16"]


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


def set_up_layers(model: str, dim_Omega: int) -> List[Module]:
    """Set up the layers of the neural network.

    Args:
        model: The name of the model. Must be in `SUPPORTED_MODELS`.
        dim_Omega: The spatial dimension of the domain Ω.

    Returns:
        A list of PyTorch modules representing the layers of the model.

    Raises:
        ValueError: If the model is not supported.
    """
    if model == "mlp-tanh-64":
        layers = [
            Linear(dim_Omega, 64),
            Tanh(),
            Linear(64, 1),
        ]
    elif model == "mlp-tanh-64-48-32-16":
        layers = [
            Linear(dim_Omega, 64),
            Tanh(),
            Linear(64, 48),
            Tanh(),
            Linear(48, 32),
            Tanh(),
            Linear(32, 16),
            Tanh(),
            Linear(16, 1),
        ]
    else:
        raise ValueError(
            f"Unsupported model: {model}. Supported models: {SUPPORTED_MODELS}"
        )

    return layers


def main():
    """Execute training with the specified command line arguments."""
    args = parse_general_args(verbose=True)

    dev = device("cuda" if cuda.is_available() else "cpu")
    dt = args.dtype
    print(f"Running on device {str(dev)}")

    manual_seed(args.data_seed)

    X_Omega = rand(args.N_Omega, args.dim_Omega).to(dev, dt)
    X_Omega_eval = rand(10 * args.N_Omega, args.dim_Omega).to(dev, dt)
    X_dOmega = poisson_equation.square_boundary(args.N_dOmega, args.dim_Omega).to(
        dev, dt
    )

    if args.equation == "poisson":
        f = poisson_equation.f
        u = poisson_equation.u

    elif args.equation == "poisson_cos_sum":
        f = poisson_equation.f_cos_sum
        u = poisson_equation.u_cos_sum

    y_Omega = f(X_Omega)
    y_dOmega = u(X_dOmega)

    # neural net
    manual_seed(args.model_seed)
    layers = set_up_layers(args.model, args.dim_Omega)
    layers = [layer.to(dev, dt) for layer in layers]
    model = Sequential(*layers).to(dev)
    print(f"Model: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # optimizer and its hyper-parameters
    optimizer, optimizer_args = set_up_optimizer(layers, args.optimizer, verbose=True)
    check_all_args_parsed()

    if args.wandb:
        config = vars(args) | vars(optimizer_args)
        wandb.init(config=config)

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
        elif isinstance(optimizer, LBFGS):
            # LBFGS requires a closure

            def closure() -> Tensor:
                """Evaluate the loss on the current data and model parameters.

                Returns:
                    The loss.
                """
                optimizer.zero_grad()
                # compute the interior loss' gradient
                loss_interior, _, _ = evaluate_interior_loss(layers, X_Omega, y_Omega)
                loss_interior.backward()

                # compute the boundary loss' gradient
                loss_boundary, _, _ = evaluate_boundary_loss(layers, X_dOmega, y_dOmega)
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

                Returns:
                    The linearization point and the loss.
                """
                loss_interior, residual_interior, _ = evaluate_interior_loss(
                    layers, X_Omega, y_Omega
                )
                loss_boundary, residual_boundary, _ = evaluate_boundary_loss(
                    layers, X_dOmega, y_dOmega
                )
                # we want to linearize residual w.r.t. the parameters to obtain
                # the GGN. This established the connection between the loss and
                # the concatenated boundary and interior residuals.
                residual = cat([residual_interior, residual_boundary])
                loss = 0.5 * (residual**2).mean()

                # HOTFIX Append the interior and boundary loss to loss_storage
                # so we can extract them for logging and plotting
                loss_storage.append((loss_interior.detach(), loss_boundary.detach()))

                return loss, residual

            forward = partial(forward, loss_storage=loss_storage)
            optimizer.step(forward)
            loss_interior, loss_boundary = loss_storage[0]

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
            l2 = l2_error(model, X_Omega_eval, u)
            print(
                f"Step: {step:06g}/{args.num_steps:06g},"
                + f" Loss: {loss},"
                + f" L2 Error: {l2},"
                + f" Interior: {loss_interior:.10f},"
                + f" Boundary: {loss_boundary:.10f},"
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
                        "l2_error": l2,
                        "time": expired,
                    }
                )


if __name__ == "__main__":
    main()
