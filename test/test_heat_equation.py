"""Test functionality for solving the heat equation."""

from test.utils import report_nonclose

from einops import einsum
from pytest import mark
from torch import allclose, cat, manual_seed, rand, zeros, zeros_like
from torch.autograd import grad
from torch.nn import Linear, Sequential, Tanh

from kfac_pinns_exp.autodiff_utils import (
    autograd_input_hessian,
    autograd_input_jacobian,
)
from kfac_pinns_exp.heat_equation import (
    evaluate_boundary_loss,
    evaluate_interior_loss,
    square_boundary_random_time,
    u_sin_product,
    unit_square_at_start,
)

DIM_OMEGAS = [1, 3]
DIM_OMEGA_IDS = [f"dim_Omega={dim}" for dim in DIM_OMEGAS]


@mark.parametrize("dim_Omega", DIM_OMEGAS, ids=DIM_OMEGA_IDS)
def test_evaluate_interior_loss(dim_Omega: int):
    """Check that autograd and Taylor-mode implementation of interior loss match.

    Args:
        dim_Omega: The spatial dimension of the domain.
    """
    manual_seed(0)
    layers = [
        Linear(dim_Omega + 1, 4),
        Tanh(),
        Linear(4, 3),
        Tanh(),
        Linear(3, 2),
        Tanh(),
        # last layer bias affects neither the spatial Laplacian, nor the time Jacobian.
        # If we enable it, we must set `allow_unused=True` and `materialize_grads=True`
        # in the below calls to `torch.autograd.grad`.
        Linear(2, 1, bias=False),
    ]
    model = Sequential(*layers)
    params = list(model.parameters())
    batch_size = 5
    X = rand(batch_size, dim_Omega + 1)
    y = zeros(batch_size, 1)

    # compute via Sequential (using autograd)
    loss_auto, residual_auto, _ = evaluate_interior_loss(model, X, y)
    grad_auto = grad(loss_auto, params)

    # compute via layers (using Taylor-mode)
    loss_taylor, residual_taylor, _ = evaluate_interior_loss(layers, X, y)
    grad_taylor = grad(loss_taylor, params)

    report_nonclose(residual_auto, residual_taylor)
    assert not allclose(residual_auto, zeros_like(residual_auto))
    report_nonclose(loss_auto, loss_taylor)
    assert not allclose(loss_auto, zeros_like(loss_auto))
    for g_auto, g_taylor in zip(grad_auto, grad_taylor):
        report_nonclose(g_auto, g_taylor)
        assert not allclose(g_auto, zeros_like(g_auto))


@mark.parametrize("dim_Omega", DIM_OMEGAS, ids=DIM_OMEGA_IDS)
def test_evaluate_boundary_loss(dim_Omega: int):
    """Check that autograd and manual implementation of condition loss match.

    Args:
        dim_Omega: The spatial dimension of the domain.
    """
    manual_seed(0)
    dim_Omega = 1
    layers = [
        Linear(dim_Omega + 1, 4),
        Tanh(),
        Linear(4, 3),
        Tanh(),
        Linear(3, 2),
        Tanh(),
        # last layer bias affects neither the spatial Laplacian, nor the time Jacobian.
        # If we enable it, we must set `allow_unused=True` and `materialize_grads=True`
        # in the below calls to `torch.autograd.grad`.
        Linear(2, 1, bias=False),
    ]
    model = Sequential(*layers)
    params = list(model.parameters())
    batch_size = 10
    X_boundary = square_boundary_random_time(batch_size // 2, dim_Omega)
    y_boundary = zeros(batch_size // 2, 1)
    X_initial = unit_square_at_start(batch_size // 2, dim_Omega)
    y_initial = u_sin_product(X_initial)
    X = cat([X_boundary, X_initial])
    y = cat([y_boundary, y_initial])

    # compute via Sequential (using autograd)
    loss_auto, residual_auto, _ = evaluate_boundary_loss(model, X, y)
    grad_auto = grad(loss_auto, params)

    # compute via layers (using Taylor-mode)
    loss_taylor, residual_taylor, _ = evaluate_boundary_loss(layers, X, y)
    grad_taylor = grad(loss_taylor, params)

    report_nonclose(residual_auto, residual_taylor)
    assert not allclose(residual_auto, zeros_like(residual_auto))
    report_nonclose(loss_auto, loss_taylor)
    assert not allclose(loss_auto, zeros_like(loss_auto))
    for g_auto, g_taylor in zip(grad_auto, grad_taylor):
        report_nonclose(g_auto, g_taylor)
        assert not allclose(g_auto, zeros_like(g_auto))


@mark.parametrize("dim_Omega", DIM_OMEGAS, ids=DIM_OMEGA_IDS)
def test_u_sin_product(dim_Omega: int):
    """Test that the sine product solution satisfies the heat equation.

    Args:
        dim_Omega: The spatial dimension of the domain.
    """
    num_data_total = 30
    # points from the interior
    X_interior = rand(num_data_total // 3, dim_Omega + 1)
    # points from the boundary conditions
    X_boundary = square_boundary_random_time(num_data_total // 3, dim_Omega)
    # points from the initial condition
    X_initial = unit_square_at_start(num_data_total // 3, dim_Omega)

    for X in [X_interior, X_boundary, X_initial]:
        input_hessian = autograd_input_hessian(u_sin_product, X)[:, 1:][:, :, 1:]
        input_laplacian = einsum(input_hessian, "batch i i -> batch").unsqueeze(-1)
        time_jacobian = autograd_input_jacobian(u_sin_product, X)[:, :, 0]
        assert input_laplacian.shape == time_jacobian.shape
        report_nonclose(input_laplacian / 4, time_jacobian)
        assert not allclose(input_laplacian, zeros_like(input_laplacian))
