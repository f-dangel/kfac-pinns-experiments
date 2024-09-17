"""Diffusivity, vector fields, and solutions of an isotropic Fokker-Planck equation."""

from functools import partial
from math import sqrt

from torch import Tensor, eye, ones, zeros
from torch.distributions import Independent, Normal

from kfac_pinns_exp import fokker_planck_equation, log_fokker_planck_equation


def mu_isotropic(x: Tensor) -> Tensor:
    """Isotropic vector field.

    Args:
        x: Un-batched input of shape `(1 + dim_Omega)` containing time and spatial
            coordinates, or batched input of shape `(batch_size, 1 + dim_Omega)`.

    Returns:
        The vector field as tensor of shape `(dim_Omega)`, or `(batch_size, dim_Omega)`
        if `x` is batched.
    """
    dim_Omega = x.shape[-1] - 1
    _, spatial = x.split([1, dim_Omega], dim=-1)
    return -0.5 * spatial


def div_mu_isotropic(x: Tensor) -> Tensor:
    """Divergence of the isotropic vector field.

    Args:
        x: Un-batched input of shape `(1 + dim_Omega)` containing time and spatial
            coordinates, or batched input of shape `(batch_size, 1 + dim_Omega)`.

    Returns:
        The vector field's divergence as tensor of shape `(1,)`, or `(batch_size, 1)`
        if `x` is batched.
    """
    dim_Omega = x.shape[-1] - 1
    return -0.5 * dim_Omega * ones(x.shape[:-1] + (1,), dtype=x.dtype, device=x.device)


def sigma_isotropic(X: Tensor) -> Tensor:
    """Isotropic diffusivity matrix.

    Args:
        X: Batched input of shape `(batch_size, 1 + dim_Omega)` containing time and
            spatial coordinates.

    Returns:
        The diffusivity matrix as tensor of shape `(batch_size, dim_Omega, dim_Omega)`.
    """
    (batch_size, dim) = X.shape
    dim_Omega = dim - 1
    return (
        sqrt(2) * eye(dim_Omega, dtype=X.dtype, device=X.device).unsqueeze(0)
    ).expand(batch_size, dim_Omega, dim_Omega)


def q_isotropic_gaussian(X: Tensor) -> Tensor:
    """Isotropic Gaussian solution to the Fokker-Planck equation in log-space.

    Args:
        X: Batched quadrature points of shape `(N, d_Omega + 1)`.

    Returns:
        The function values as tensor of shape `(N, 1)`.
    """
    batch_size, d = X.shape
    d -= 1

    exp_t = (-X[:, 0]).exp()
    covariance = exp_t + 2 * (1 - exp_t)  # [batch_size]
    # [batch_size, d]
    mean = zeros(batch_size, d, device=X.device, dtype=X.dtype)
    std = covariance.unsqueeze(1).expand(-1, d).sqrt()

    # NOTE Normal wants the standard deviation, not the covariance
    base_dist = Normal(mean, std)
    dist = Independent(base_dist, 1)
    output = dist.log_prob(X[:, 1:])

    return output.unsqueeze(-1)


evaluate_interior_loss = partial(
    log_fokker_planck_equation.evaluate_interior_loss,
    mu=mu_isotropic,
    sigma=sigma_isotropic,
    div_mu=div_mu_isotropic,
    sigma_isotropic=True,
)

evaluate_interior_loss_and_kfac = partial(
    log_fokker_planck_equation.evaluate_interior_loss_and_kfac,
    mu=mu_isotropic,
    sigma=sigma_isotropic,
    div_mu=div_mu_isotropic,
    sigma_isotropic=True,
)

evaluate_interior_loss_with_layer_inputs_and_grad_outputs = partial(
    log_fokker_planck_equation.evaluate_interior_loss_with_layer_inputs_and_grad_outputs,  # noqa: B950
    mu=mu_isotropic,
    sigma=sigma_isotropic,
    div_mu=div_mu_isotropic,
    sigma_isotropic=True,
)

evaluate_interior_loss_with_layer_inputs_and_grad_outputs = partial(
    log_fokker_planck_equation.evaluate_interior_loss_with_layer_inputs_and_grad_outputs,  # noqa: B950
    mu=mu_isotropic,
    sigma=sigma_isotropic,
    div_mu=div_mu_isotropic,
)

plot_solution = partial(
    fokker_planck_equation.plot_solution, solutions={"gaussian": q_isotropic_gaussian}
)
