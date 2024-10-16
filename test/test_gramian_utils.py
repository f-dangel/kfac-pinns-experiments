"""Test `kfac_pinns_exp.gramian_utils`."""

from typing import List, Union

from pytest import mark
from torch import Tensor, allclose, cat, manual_seed, outer, rand, zeros, zeros_like
from torch.autograd import grad
from torch.nn import Linear, Module, Sequential, Tanh

from kfac_pinns_exp import (
    fokker_planck_isotropic_equation,
    log_fokker_planck_isotropic_equation,
)
from kfac_pinns_exp.gramian_utils import autograd_gramian

LOSS_TYPES = [
    "poisson_boundary",
    "poisson_interior",
    "heat_boundary",
    "heat_interior",
    "fokker-planck-isotropic_boundary",
    "fokker-planck-isotropic_interior",
    "log-fokker-planck-isotropic_boundary",
    "log-fokker-planck-isotropic_interior",
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
            `'fokker-planck-isotropic_boundary'`,
            `'log-fokker-planck-isotropic_interior'`, or
            `'log-fokker-planck-isotropic_boundary'`.
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
            in {
                "poisson_interior",
                "heat_interior",
                "fokker-planck-isotropic_interior",
                "log-fokker-planck-isotropic_interior",
            }
        )
        output = model(X_n)

        if loss_type in {
            "poisson_boundary",
            "heat_boundary",
            "fokker-planck-isotropic_boundary",
            "log-fokker-planck-isotropic_boundary",
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
        elif loss_type == "log-fokker-planck-isotropic_interior":
            q = model(X_n)

            # compute dq/dt
            (dq,) = grad(q, X_n, create_graph=True)
            dq_dt, dq_dX = dq[0], dq[1:]

            # compute div(μ)
            mu_X = log_fokker_planck_isotropic_equation.mu_isotropic(X_n)
            div_mu = 0
            for d in range(1, D_in):
                (jac,) = grad(mu_X[d - 1], X_n, create_graph=True)
                div_mu += jac[d]

            # compute (∇ₓq)ᵀ μ
            dq_dX_mu = (dq_dX * mu_X).sum()

            # compute || σᵀ ∇ₓq ||²
            sigma = log_fokker_planck_isotropic_equation.sigma_isotropic(
                X_n.unsqueeze(0)
            ).squeeze(0)
            norm_sigma_dq_dX = (sigma.T @ dq_dX).norm() ** 2

            # spatial Hessian
            hess = zeros((D_in, D_in), device=X_n.device, dtype=X_n.dtype)
            for d in range(D_in):
                (grad_input,) = grad(q, X_n, create_graph=True)
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
                dq_dt
                + div_mu
                + dq_dX_mu
                - 0.5 * norm_sigma_dq_dX
                - 0.5 * tr_sigma_outer_hess,
                params,
                retain_graph=True,
                # last layer bias does not contribute to the differential operator
                allow_unused=True,
                materialize_grads=True,
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
