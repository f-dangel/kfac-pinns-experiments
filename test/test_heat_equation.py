"""Test functionality for solving the heat equation."""

from test.utils import report_nonclose

from einops import einsum
from pytest import mark
from torch import allclose, cat, manual_seed, rand, tensor, zeros, zeros_like
from torch.autograd import grad
from torch.nn import Linear, Sequential, Tanh

from kfac_pinns_exp.autodiff_utils import (
    autograd_input_hessian,
    autograd_input_jacobian,
)
from kfac_pinns_exp.heat_equation import (
    evaluate_boundary_loss,
    evaluate_boundary_loss_and_kfac,
    evaluate_interior_loss,
    evaluate_interior_loss_and_kfac,
    square_boundary_random_time,
    u_sin_product,
    u_sin_sum,
    unit_square_at_start,
)

DIM_OMEGAS = [1, 3]
DIM_OMEGA_IDS = [f"dim_Omega={dim}" for dim in DIM_OMEGAS]


@mark.parametrize("dim_Omega", DIM_OMEGAS, ids=DIM_OMEGA_IDS)
def test_evaluate_interior_loss(dim_Omega: int):
    """Check that autograd and forward Laplacian implementation of interior loss match.

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

    # compute via layers (using forward Laplacian)
    loss_manual, residual_manual, _ = evaluate_interior_loss(layers, X, y)
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


@mark.parametrize("condition", ["sin_product", "sin_sum"], ids=str)
@mark.parametrize("dim_Omega", DIM_OMEGAS, ids=DIM_OMEGA_IDS)
def test_heat_equation_solutions(dim_Omega: int, condition: str):
    """Test that a manual solution satisfies the heat equation.

    Args:
        dim_Omega: The spatial dimension of the domain.
        condition: The type of condition.
    """
    num_data_total = 30
    # points from the interior
    X_interior = rand(num_data_total // 3, dim_Omega + 1)
    # points from the boundary conditions
    X_boundary = square_boundary_random_time(num_data_total // 3, dim_Omega)
    # points from the initial condition
    X_initial = unit_square_at_start(num_data_total // 3, dim_Omega)

    coordinates = list(range(1, dim_Omega + 1))
    u = {"sin_product": u_sin_product, "sin_sum": u_sin_sum}[condition]

    for X in [X_interior, X_boundary, X_initial]:
        input_hessian = autograd_input_hessian(u, X, coordinates=coordinates)
        input_laplacian = einsum(input_hessian, "batch i i -> batch").unsqueeze(-1)
        time_jacobian = autograd_input_jacobian(u, X)[:, :, 0]
        assert input_laplacian.shape == time_jacobian.shape
        report_nonclose(input_laplacian / 4, time_jacobian)
        assert not allclose(input_laplacian, zeros_like(input_laplacian))


def test_evaluate_interior_loss_and_kfac():
    """Make sure the interior KFAC computation does not change over time.

    We computed KFAC on a commit we trust, because the related optimizer showed good
    performance. We now want to make sure that the KFAC computation does not change
    over time, because it is crucial for the optimizer to work correctly.

    Commit: dbf432d6acc6d40e308500bf96d662213d0d919a
    Hardware: MacBook Pro M2, PyTorch 2.2.0, Python 3.9.16
    """
    manual_seed(0)
    dim_Omega = 2
    layers = [Linear(dim_Omega + 1, 4), Tanh(), Linear(4, 3), Tanh(), Linear(3, 1)]
    tols = {"rtol": 2e-5}
    N = 20
    X, y = rand(N, dim_Omega + 1), zeros(N, 1)

    loss, kfacs = evaluate_interior_loss_and_kfac(layers, X, y)
    assert list(kfacs.keys()) == [0, 2, 4]
    report_nonclose(loss * 10**6, tensor(2.80977), **tols)

    # first Linear KFAC
    A_0 = tensor(
        [
            [2.54571e-01, 5.63837e-02, 3.72303e-02, 8.43531e-02],
            [5.63837e-02, 2.88833e-01, 6.35285e-02, 1.22396e-01],
            [3.72303e-02, 6.35285e-02, 2.73013e-01, 1.04975e-01],
            [8.43531e-02, 1.22396e-01, 1.04975e-01, 2.00000e-01],
        ]
    )
    B_0 = tensor(
        [
            [8.45101e-04, -5.56851e-04, -5.91830e-03, -2.02141e-03],
            [-5.56851e-04, 4.26722e-04, 4.26222e-03, 1.58313e-03],
            [-5.91830e-03, 4.26222e-03, 4.62377e-02, 1.55846e-02],
            [-2.02141e-03, 1.58313e-03, 1.55846e-02, 6.03931e-03],
        ]
    )
    report_nonclose(kfacs[0][0], A_0, **tols)
    report_nonclose(kfacs[0][1], B_0, **tols)

    # second Linear KFAC
    A_2 = tensor(
        [
            [1.03467e-01, 4.45037e-02, 2.33129e-02, 9.44624e-03, -1.07210e-01],
            [4.45037e-02, 8.66871e-02, -1.73244e-02, 1.73515e-03, -1.07309e-01],
            [2.33129e-02, -1.73244e-02, 4.46495e-02, -1.55726e-02, 2.10115e-03],
            [9.44624e-03, 1.73515e-03, -1.55726e-02, 1.53613e-02, -1.59915e-02],
            [-1.07210e-01, -1.07309e-01, 2.10115e-03, -1.59915e-02, 2.00000e-01],
        ]
    )
    B_2 = tensor(
        [
            [1.46475e-01, -1.18647e-01, -1.00829e-01],
            [-1.18647e-01, 9.66549e-02, 8.24751e-02],
            [-1.00829e-01, 8.24751e-02, 7.17428e-02],
        ]
    )
    report_nonclose(kfacs[2][0], A_2, **tols)
    report_nonclose(kfacs[2][1], B_2, **tols)

    # fourth Linear KFAC
    A_4 = tensor(
        [
            [5.67683e-02, 4.94787e-02, 1.09211e-02, -1.02528e-01],
            [4.94787e-02, 4.40693e-02, 6.63877e-03, -8.87352e-02],
            [1.09211e-02, 6.63877e-03, 1.99315e-02, -2.73296e-02],
            [-1.02528e-01, -8.87352e-02, -2.73296e-02, 2.00000e-01],
        ]
    )
    B_4 = tensor([[1.06250e00]])
    report_nonclose(kfacs[4][0], A_4, **tols)
    report_nonclose(kfacs[4][1], B_4, **tols)


def test_evaluate_boundary_loss_and_kfac():
    """Make sure the boundary KFAC computation does not change over time.

    We computed KFAC on a commit we trust, because the related optimizer showed good
    performance. We now want to make sure that the KFAC computation does not change
    over time, because it is crucial for the optimizer to work correctly.

    Commit: dbf432d6acc6d40e308500bf96d662213d0d919a
    Hardware: MacBook Pro M2, PyTorch 2.2.0, Python 3.9.16
    """
    manual_seed(0)
    dim_Omega = 2
    layers = [Linear(dim_Omega + 1, 4), Tanh(), Linear(4, 3), Tanh(), Linear(3, 1)]
    tols = {"rtol": 2e-5}
    N = 10
    X_boundary = square_boundary_random_time(N // 2, dim_Omega)
    y_boundary = zeros(N // 2, 1)
    X_initial = unit_square_at_start(N // 2, dim_Omega)
    y_initial = u_sin_product(X_initial)
    X = cat([X_boundary, X_initial])
    y = cat([y_boundary, y_initial])

    loss, kfacs = evaluate_boundary_loss_and_kfac(layers, X, y)
    assert list(kfacs.keys()) == [0, 2, 4]
    report_nonclose(loss, tensor(3.21899e-1), **tols)

    # first Linear KFAC
    A_0 = tensor(
        [
            [1.98280e-01, 1.04688e-01, 9.58195e-02, 2.63509e-01],
            [1.04688e-01, 3.19806e-01, 2.95681e-01, 4.85086e-01],
            [9.58195e-02, 2.95681e-01, 4.50549e-01, 5.62863e-01],
            [2.63509e-01, 4.85086e-01, 5.62863e-01, 1.00000e00],
        ]
    )
    B_0 = tensor(
        [
            [6.96504e-04, -5.33558e-04, -5.46882e-03, -2.00172e-03],
            [-5.33558e-04, 4.23520e-04, 4.30012e-03, 1.59442e-03],
            [-5.46882e-03, 4.30012e-03, 4.40618e-02, 1.61296e-02],
            [-2.00172e-03, 1.59442e-03, 1.61296e-02, 6.05028e-03],
        ]
    )
    report_nonclose(kfacs[0][0], A_0, **tols)
    report_nonclose(kfacs[0][1], B_0, **tols)

    # second Linear KFAC
    A_2 = tensor(
        [
            [3.43695e-01, 2.54821e-01, 3.37492e-02, 5.39269e-02, -5.73159e-01],
            [2.54821e-01, 2.29415e-01, 1.88713e-02, 3.14351e-02, -4.62159e-01],
            [3.37492e-02, 1.88713e-02, 1.89985e-02, -4.25742e-03, -4.68395e-02],
            [5.39269e-02, 3.14351e-02, -4.25742e-03, 1.83925e-02, -8.59532e-02],
            [-5.73159e-01, -4.62159e-01, -4.68395e-02, -8.59532e-02, 1.00000e00],
        ]
    )
    B_2 = tensor(
        [
            [1.44919e-01, -1.17004e-01, -9.78227e-02],
            [-1.17004e-01, 9.45760e-02, 7.89963e-02],
            [-9.78227e-02, 7.89963e-02, 6.62688e-02],
        ]
    )
    report_nonclose(kfacs[2][0], A_2, **tols)
    report_nonclose(kfacs[2][1], B_2, **tols)

    # third Linear KFAC
    A_4 = tensor(
        [
            [2.37701e-01, 2.05744e-01, 7.33859e-02, -4.86125e-01],
            [2.05744e-01, 1.78799e-01, 6.13888e-02, -4.21101e-01],
            [7.33859e-02, 6.13888e-02, 3.19807e-02, -1.51726e-01],
            [-4.86125e-01, -4.21101e-01, -1.51726e-01, 1.00000e00],
        ]
    )
    B_4 = tensor([[1.00000e00]])
    report_nonclose(kfacs[4][0], A_4, **tols)
    report_nonclose(kfacs[4][1], B_4, **tols)
