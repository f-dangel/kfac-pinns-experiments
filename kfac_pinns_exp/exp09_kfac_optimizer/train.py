"""Train with KFAC for the Poisson equation."""

from argparse import ArgumentParser, Namespace
from math import pi
from time import time

from torch import Tensor, cos, cuda, device, manual_seed, prod, rand, randint
from torch.nn import Linear, Sequential, Tanh

from kfac_pinns_exp.exp09_kfac_optimizer.optimizer import KFACForPINNs


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        A namespace with the parsed arguments.
    """
    parser = ArgumentParser()

    parser.add_argument(
        "--dim_Omega",
        type=int,
        default=2,
        help="Spatial dimension of the Poisson equation's domain Ω.",
    )
    parser.add_argument(
        "--N_Omega",
        type=int,
        default=1_000,
        help="Number of quadrature points in the domain Ω.",
    )
    parser.add_argument(
        "--N_dOmega",
        type=int,
        default=100,
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

    return parser.parse_args()


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

    # infer spatial dimension of x
    d = Tensor([len(X[0])])

    return d * pi**2 * prod(cos(pi * X), dim=1, keepdim=True)


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
    args = parse_args()

    # Spatial dimension and sampling points, i.e., batch dimensions.
    print("Poisson equation")
    print(f"dim(Ω) = {args.dim_Omega}")
    print(f"N(Ω) = {args.N_Omega}")
    print(f"N(∂Ω) = {args.N_dOmega}")

    dev = device("cuda" if cuda.is_available() else "cpu")

    manual_seed(args.data_seed)
    # quadrature points = data
    # interior
    X_Omega = rand(args.N_Omega, args.dim_Omega).to(dev)
    y_Omega = -f(X_Omega)
    # boundary
    X_dOmega = square_boundary(args.N_dOmega, args.dim_Omega).to(dev)
    y_dOmega = u(X_dOmega)

    # neural net
    manual_seed(args.model_seed)
    layers = [
        Linear(args.dim_Omega, 256),
        Tanh(),
        Linear(256, 64),
        Tanh(),
        Linear(64, 16),
        Tanh(),
        Linear(16, 1),
    ]
    layers = [l.to(dev) for l in layers]
    model = Sequential(*layers).to(dev)
    print(f"Model: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # optimizer hyper-parameters
    T_kfac = 2
    T_inv = 6
    ema_factor = 0.95  # exponentially moving average over KFAC factors
    damping = 1e-2
    lr = 1e-1

    optimizer = KFACForPINNs(
        layers, lr, damping, T_kfac=T_kfac, T_inv=T_inv, ema_factor=ema_factor
    )

    num_steps = 50
    print_every = 10
    start = time()
    last = start

    # training loop
    for step in range(num_steps):
        optimizer.zero_grad()

        # depending on step, either updates the KFAC approximation and the loss, or
        # only computes the loss
        loss_interior = optimizer.evaluate_interior_loss_and_update_kfac(
            X_Omega, y_Omega
        )

        # compute the interior loss' gradient
        loss_interior.backward()

        # depending on step, either updates the KFAC approximation and the loss, or
        # only computes the loss
        loss_boundary = optimizer.evaluate_boundary_loss_and_update_kfac(
            X_dOmega, y_dOmega
        )

        # compute the boundary loss' gradient
        loss_boundary.backward()

        if step % print_every == 0 or step == num_steps - 1:
            now = time()
            print(
                f"Step: {step:06g}, Loss: {(loss_interior + loss_boundary).item():.3f},"
                + f" Interior: {loss_interior.item():.3f},"
                + f" Boundary: {loss_boundary.item():.3f},"
                + f" Time: {time() - start:.1f}s"
                + f" ({(now - last) / print_every:.2f}s/iter)"
            )
            last = now

        # depending on step, maybe update the inverse curvature, compute the
        # pre-conditioned gradient and update the parameters, increment step internally
        optimizer.step()


if __name__ == "__main__":
    main()
