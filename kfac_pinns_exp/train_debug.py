"""Script to mix ENGD and its KFAC approximation step by step for debugging.

Step by step we move the curvature matrix from the ENGD one to the KFAC one.
We check performance at every step.

1) G = G_Omega^ENGD + G_partialOmega^KFAC  <------this is what we are working on now.
2) G = G_Omega^KFAC + G_partialOmega^ENGD
3) G = G_Omega^KFAC + G_partialOmega^KFAC

"""

from argparse import ArgumentParser, Namespace
from functools import partial
from math import log10
from time import time
from typing import List, Tuple

from hessianfree.optimizer import HessianFree
from torch import (
    Tensor,
    block_diag,
    cat,
    cuda,
    device,
    float32,
    float64,
    kron,
    logspace,
    manual_seed,
    rand,
)
from torch.linalg import matrix_norm
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
        default="ENGD",
        choices=SUPPORTED_OPTIMIZERS,
        help="Which optimizer will be used.",
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
    optimizer, optimizer_args = set_up_optimizer(layers, "ENGD", verbose=True)
    optimizer_kfac = KFAC(layers=layers, damping=0.0)
    check_all_args_parsed()

    logged_steps = {
        int(s) for s in logspace(0, log10(args.num_steps - 1), args.max_logs - 1).int()
    } | {0}

    start = time()

    # training loop
    for step in range(args.num_steps):
        optimizer.zero_grad()

        ###############THE OPTIMIZER STEP UNROLLED############
        # ENGD STEP BEGINS
        # compute gradients and Gramians on current data
        interior_gramian = poisson_equation.evaluate_interior_gramian(
            optimizer.model, X_Omega, approximation="full"
        )
        boundary_gramian = poisson_equation.evaluate_boundary_gramian(
            optimizer.model, X_dOmega, approximation="full"
        )

        # KFAC BOUNDARY MATRIX
        _, kfacs = poisson_equation.evaluate_boundary_loss_and_kfac_expand(
            layers, X_dOmega, y_dOmega
        )
        kfacs_multiplied = []
        for value in kfacs.values():
            kfacs_multiplied.append(kron(value[0], value[1]))

        # I have normalized here... but I did this only because otherwise the matrix_norms would be very differently below...
        G_KFAC_BDRY = 1.0 / len(y_dOmega) * block_diag(*kfacs_multiplied)

        print(matrix_norm(boundary_gramian))
        print(matrix_norm(G_KFAC_BDRY))
        print(matrix_norm(G_KFAC_BDRY - boundary_gramian))

        # break up optimizer API
        optimizer.gramian = interior_gramian + G_KFAC_BDRY
        interior_loss, _, _ = evaluate_interior_loss(optimizer.model, X_Omega, y_Omega)
        interior_loss.backward()
        boundary_loss, _, _ = evaluate_boundary_loss(
            optimizer.model, X_dOmega, y_dOmega
        )
        boundary_loss.backward()
        natgrads = optimizer._compute_natural_gradients()
        optimizer._update_parameters(natgrads, X_Omega, y_Omega, X_dOmega, y_dOmega)
        # ENGD STEP ENDS

        now = time()
        expired = now - start
        loss = interior_loss + boundary_loss

        if step in logged_steps:
            l2 = l2_error(model, X_Omega_eval, u)
            print(
                f"Step: {step:06g}/{args.num_steps:06g},"
                + f" Loss: {loss},"
                + f" L2 Error: {l2},"
                + f" Interior: {interior_loss:.10f},"
                + f" Boundary: {boundary_loss:.10f},"
                + f" Time: {expired:.1f}s",
                flush=True,
            )
        exit()


if __name__ == "__main__":
    main()
