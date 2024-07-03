"""Test `kfac_pinns_exp.autodiff_utils`."""

from itertools import product
from test.utils import report_nonclose
from typing import List, Union

from pytest import mark
from torch import Tensor, allclose, cat, manual_seed, outer, rand, zeros, zeros_like
from torch.autograd import grad
from torch.nn import Linear, Module, Sequential, Sigmoid, Tanh

from kfac_pinns_exp import fokker_planck_isotropic_equation
from kfac_pinns_exp.autodiff_utils import autograd_gramian, autograd_input_divergence

LOSS_TYPES = [
    "poisson_boundary",
    "poisson_interior",
    "heat_boundary",
    "heat_interior",
    "fokker-planck-isotropic_boundary",
    "fokker-planck-isotropic_interior",
]
APPROXIMATIONS = ["full", "diagonal", "per_layer"]


@mark.parametrize("approximation", APPROXIMATIONS, ids=APPROXIMATIONS)
@mark.parametrize("loss_type", LOSS_TYPES, ids=LOSS_TYPES)
def test_autograd_gramian(loss_type: str, approximation: str):  # noqa: C901
    """Test `autograd_gramian`.

    Args:
        loss_type: The type of loss function whose Gramian
            is tested. Can be either `'poisson_boundary'`, `'poisson_interior`,
            `'heat_boundary`, `'heat_interior'`, `'fokker-planck-isotropic_interior`,
            or `'fokker-planck-isotropic_boundary'`.
        approximation: The type of approximation to the Gramian.
            Can be either `'full'`, `'diagonal'`, or `'per_layer'`.

    Raises:
        ValueError: If `loss_type` is not one of the allowed values.
        ValueError: If `approximation` is not one of `'full'` or `'diagonal'`.
    """
    manual_seed(0)
    # hyper-parametersj
    D_in, D_hidden, D_out = 3, 10, 1
    batch_size = 5
    assert D_out == 1

    # set up data and model
    X = rand(batch_size, D_in)
    model = Sequential(
        Linear(D_in, D_hidden),
        Tanh(),
        Linear(D_hidden, D_hidden),
        Tanh(),
        Linear(D_hidden, D_out),
    )

    # compute the boundary Gramian with functorch
    params = list(model.parameters())
    param_names = [n for n, _ in model.named_parameters()]
    gramian = autograd_gramian(
        model, X, param_names, loss_type=loss_type, approximation=approximation
    )

    # compute the Gramian naively via a for-loop and autograd
    dim = sum(p.numel() for p in params)
    truth = zeros(dim, dim)

    # compute the Gram gradient for sample n and add its contribution
    # to the Gramian
    for n in range(batch_size):
        X_n = X[n].requires_grad_(
            loss_type
            in {"poisson_interior", "heat_interior", "fokker-planck-isotropic_interior"}
        )
        output = model(X_n)

        if loss_type in {
            "poisson_boundary",
            "heat_boundary",
            "fokker-planck-isotropic_boundary",
        }:
            gram_grad = grad(output, params)

        elif loss_type == "poisson_interior":
            laplace = zeros(())

            for d in range(D_in):
                (grad_input,) = grad(output, X_n, create_graph=True)
                e_d = zeros_like(X_n)
                e_d[d] = 1.0

                (hess_input_dd,) = grad(
                    (e_d * grad_input).sum(), X_n, create_graph=True
                )
                laplace += hess_input_dd[d]

            gram_grad = grad(
                laplace,
                params,
                retain_graph=True,
                # set gradients of un-used parameters to zero
                # (e.g. last layer bias does not affect Laplacian)
                materialize_grads=True,
            )
        elif loss_type == "heat_interior":
            laplace = zeros(())
            jac = zeros(())

            # spatial Laplacian
            for d in range(1, D_in):
                (grad_input,) = grad(output, X_n, create_graph=True)
                e_d = zeros_like(X_n)
                e_d[d] = 1.0

                (hess_input_dd,) = grad(
                    (e_d * grad_input).sum(), X_n, create_graph=True
                )
                laplace += hess_input_dd[d]

            # temporal Jacobian
            (grad_input,) = grad(output, X_n, create_graph=True)
            jac += grad_input[0]

            gram_grad = grad(
                jac - laplace / 4,
                params,
                retain_graph=True,
                # set gradients of un-used parameters to zero
                # (e.g. last layer bias does not affect Laplacian)
                materialize_grads=True,
            )
        elif loss_type == "fokker-planck-isotropic_interior":
            p = model(X_n)

            # compute dp/dt
            (dp_dX,) = grad(p, X_n, create_graph=True)
            dp_dt = dp_dX[0]

            # compute div(p * μ)
            p_times_mu = p * fokker_planck_isotropic_equation.mu_isotropic(X_n)
            div_p_times_mu = 0
            for d in range(1, D_in):
                (jac,) = grad(p_times_mu[d - 1], X_n, create_graph=True)
                div_p_times_mu += jac[d]
            assert p_times_mu.ndim == 1

            # spatial Hessian
            hess = zeros((D_in, D_in), device=X_n.device, dtype=X_n.dtype)
            for d in range(D_in):
                (grad_input,) = grad(p, X_n, create_graph=True)
                e_d = zeros_like(X_n)
                e_d[d] = 1.0
                (hess_input_d,) = grad((e_d * grad_input).sum(), X_n, create_graph=True)
                hess[d] = hess_input_d
            hess = hess[1:,][:, 1:]

            # compute 0.5 * tr(σ σᵀ ∇²ₓp)
            sigma = fokker_planck_isotropic_equation.sigma_isotropic(
                X_n.unsqueeze(0)
            ).squeeze(0)
            tr_sigma_outer_hess = (sigma @ sigma.T @ hess).trace()

            gram_grad = grad(
                dp_dt + div_p_times_mu - 0.5 * tr_sigma_outer_hess,
                params,
                retain_graph=True,
            )
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # flatten and take the outer product
        gram_grad = cat([g.flatten() for g in gram_grad])
        truth.add_(outer(gram_grad, gram_grad))

    truth = extract_approximation(truth, model, approximation)

    if approximation == "per_layer":
        for b in range(len(truth)):
            assert allclose(gramian[b], truth[b])
    elif approximation in ["diagonal", "full"]:
        assert allclose(gramian, truth)


def extract_approximation(
    gramian: Tensor, model: Module, approximation: str
) -> Union[Tensor, List[Tensor]]:
    """Extract the desired approximation from the Gramian.

    Args:
        gramian: The Gramian matrix.
        model: The model whose Gramian is computed.
        approximation: The type of approximation to the Gramian.
            Can be either `'full'`, `'diagonal'`, or `'per_layer'`.

    Returns:
        The desired approximation to the Gramian.

    Raises:
        ValueError: If `approximation` is not one of `'full'`, `'diagonal'`,
            or `'per_layer'`.
    """
    # account for approximation
    if approximation == "diagonal":
        return gramian.diag()
    elif approximation == "full":
        return gramian
    elif approximation == "per_layer":
        sizes = [
            sum(p.numel() for p in layer.parameters())
            for layer in model.modules()
            if not list(layer.children()) and list(layer.parameters())
        ]
        # cut the Gramian into per_layer blocks
        gramian = [
            row_block.split(sizes, dim=1) for row_block in gramian.split(sizes, dim=0)
        ]
        return [gramian[b][b] for b in range(len(sizes))]

    raise ValueError(f"Unknown approximation: {approximation}.")


def test_autograd_input_divergence():
    """Test computation of the divergence with `functorch`."""
    manual_seed(0)

    # setup
    N, S, D = 5, 4, 3
    X = rand(N, S, D, requires_grad=True)
    X.requires_grad_(True)
    model = Sequential(Linear(D, D), Tanh(), Linear(D, D), Sigmoid())

    # compute the divergence with `autograd`
    div_true = zeros(N, 1)

    f_X = model(X)

    for n, d, s in product(range(N), range(D), range(S)):
        e = zeros_like(f_X)
        e[n, s, d] = 1.0
        div_true[n] += grad(f_X, X, grad_outputs=e, retain_graph=True)[0][n, s, d]

    # compute the divergence with `functorch`
    div = autograd_input_divergence(model, X)

    assert div.shape == div_true.shape == (N, 1)
    report_nonclose(div, div_true)


def test_autograd_input_divergence_with_coordinates():
    """Test computation of the divergence with specified coordinates."""
    manual_seed(0)

    # setup
    N, S, D = 5, 4, 3
    X = rand(N, S, D, requires_grad=True)

    # S * D input coordinates, but only S output coordinates
    model = Sequential(Linear(D, D), Tanh(), Linear(D, 1), Sigmoid())

    # compute the divergence with `autograd`
    div_true = zeros(N, 1)

    f_X = model(X)
    coordinates = [0, 5, 3, 8]  # must be < S * D and contain S entries
    assert len(coordinates) == f_X.shape[1:].numel()

    for n in range(N):
        for out, coord in zip(range(len(coordinates)), coordinates):
            e = zeros_like(f_X).flatten(start_dim=1)
            e[n, out] = 1.0
            e = e.reshape_as(f_X)
            div_true[n] += grad(f_X, X, grad_outputs=e, retain_graph=True)[0].flatten(
                start_dim=1
            )[n, coord]

    # compute the divergence with `functorch`
    div = autograd_input_divergence(model, X, coordinates=coordinates)

    assert div.shape == div_true.shape == (N, 1)
    report_nonclose(div, div_true)
