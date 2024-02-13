"""Train with KFAC for the Poisson equation."""

from math import pi

from torch import Tensor, cos, prod, rand, randint
from torch.nn import Linear, Sequential, Tanh

from kfac_pinns_exp.exp09_kfac_optimizer.optimizer import KFACForPINNs

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


# Spatial dimension and sampling points, i.e., batch dimensions.
dim_Omega = 2
N_Omega = 1_000
N_dOmega = 100

# quadrature points = data
# interior
X_Omega = rand(N_Omega, dim_Omega)
y_Omega = -f(X_Omega)
# boundary
X_dOmega = square_boundary(N_dOmega, dim_Omega)
y_dOmega = u(X_dOmega)

# neural net
layers = [
    Linear(dim_Omega, 32),
    Tanh(),
    Linear(32, 1),
]
model = Sequential(*layers)

# optimizer hyper-parameters
T_kfac = 1
T_inv = 1
ema_factor = 0.95  # exponentially moving average over KFAC factors
damping = 1e-3
lr = 1e-2

optimizer = KFACForPINNs(
    layers, lr, damping, T_kfac=T_kfac, T_inv=T_inv, ema_factor=ema_factor
)

num_steps = 1_000

# training loop
for step in range(num_steps):
    optimizer.zero_grad()

    # depending on step, either updates the KFAC approximation and the loss, or
    # only computes the loss
    loss_interior = optimizer.evaluate_interior_loss_and_update_kfac(X_Omega, y_Omega)

    # compute the interior loss' gradient
    loss_interior.backward()

    # depending on step, either updates the KFAC approximation and the loss, or
    # only computes the loss
    loss_boundary = optimizer.evaluate_boundary_loss_and_update_kfac(X_dOmega, y_dOmega)

    # compute the boundary loss' gradient
    loss_boundary.backward()

    print(
        f"Step: {step}, Loss: {(loss_interior + loss_boundary).item()}, "
        + f"Interior: {loss_interior.item()}, Boundary: {loss_boundary.item()}"
    )

    # depending on step, maybe update the inverse curvature, compute the
    # pre-conditioned gradient and update the parameters, increment step internally
    optimizer.step()
