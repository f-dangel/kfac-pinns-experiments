"""Test functionality for solving the logarithmic Fokker-Planck equation."""

from test.utils import report_nonclose

from pytest import mark
from torch import allclose, cat, manual_seed, rand, zeros, zeros_like
from torch.autograd import grad
from torch.nn import Linear, Sequential, Tanh

from kfac_pinns_exp.log_fokker_planck_equation import (
    evaluate_boundary_loss,
    evaluate_interior_loss,
)
from kfac_pinns_exp.log_fokker_planck_isotropic_equation import (
    mu_isotropic,
    q_isotropic_gaussian,
    sigma_isotropic,
)

DIM_OMEGAS = [1, 3]
DIM_OMEGA_IDS = [f"dim_Omega={dim}" for dim in DIM_OMEGAS]


@mark.parametrize("dim_Omega", DIM_OMEGAS, ids=DIM_OMEGA_IDS)
def test_evaluate_interior_loss(dim_Omega: int):
    """Check that autograd and manual implementation of interior loss match.

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
        # Last-layer bias does not contribute to the differential operator,
        # therefore we exclude it from the architecture.
        Linear(2, 1, bias=False),
    ]
    model = Sequential(*layers)
    params = list(model.parameters())
    batch_size = 10

    t = rand(batch_size, 1)
    spatial = 10 * rand(batch_size, dim_Omega) - 5
    X = cat([t, spatial], dim=1)
    y = zeros(batch_size, 1)

    # compute via Sequential (using autograd)
    loss_auto, residual_auto, _ = evaluate_interior_loss(
        model, X, y, mu_isotropic, sigma_isotropic
    )
    grad_auto = grad(loss_auto, params)

    # compute via layers (using manual forward)
    loss_manual, residual_manual, _ = evaluate_interior_loss(
        layers, X, y, mu_isotropic, sigma_isotropic
    )
    grad_manual = grad(loss_manual, params)

    report_nonclose(residual_auto, residual_manual)
    assert not allclose(residual_auto, zeros_like(residual_auto))
    report_nonclose(loss_auto, loss_manual)
    assert not allclose(loss_auto, zeros_like(loss_auto))
    for g_auto, g_manual in zip(grad_auto, grad_manual):
        report_nonclose(g_auto, g_manual)
        assert not allclose(g_auto, zeros_like(g_auto))


@mark.parametrize("dim_Omega", DIM_OMEGAS, ids=DIM_OMEGA_IDS)
def test_evaluate_boundary_loss(dim_Omega: int):
    """Check that autograd and manual implementation of condition loss match.

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
        Linear(2, 1),
    ]
    model = Sequential(*layers)
    params = list(model.parameters())
    batch_size = 10

    X_no_t = 10 * rand(batch_size, dim_Omega) - 5
    t = zeros(batch_size, 1)
    X = cat([t, X_no_t], dim=1)
    y = q_isotropic_gaussian(X)

    # compute via Sequential (using autograd)
    loss_auto, residual_auto, _ = evaluate_boundary_loss(model, X, y)
    grad_auto = grad(loss_auto, params)

    # compute via layers (using manual forward)
    loss_manual, residual_manual, _ = evaluate_boundary_loss(layers, X, y)
    grad_manual = grad(loss_manual, params)

    report_nonclose(residual_auto, residual_manual)
    assert not allclose(residual_auto, zeros_like(residual_auto))
    report_nonclose(loss_auto, loss_manual)
    assert not allclose(loss_auto, zeros_like(loss_auto))
    for g_auto, g_manual in zip(grad_auto, grad_manual):
        report_nonclose(g_auto, g_manual)
        assert not allclose(g_auto, zeros_like(g_auto))
