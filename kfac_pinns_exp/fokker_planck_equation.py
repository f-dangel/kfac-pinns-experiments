"""Implements functionality to support solving the Fokker-Planck equation."""

from math import sqrt
from typing import Callable, Dict, List, Tuple, Union
from warnings import warn

from einops import einsum
from torch import Tensor, cat, eye, ones, zeros, zeros_like
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import Module, Sequential

from kfac_pinns_exp.autodiff_utils import (
    autograd_input_divergence,
    autograd_input_hessian,
)
from kfac_pinns_exp.manual_differentiation import manual_forward


def evaluate_interior_loss(
    model: Union[Module, List[Module]],
    X: Tensor,
    y: Tensor,
    mu: Callable[[Tensor], Tensor],
    sigma: Callable[[Tensor], Tensor],
) -> Tuple[Tensor, Tensor, Union[List[Dict[str, Tensor]], None]]:
    """Evaluate the interior loss.

    Args:
        model: The model or a list of layers that form the sequential model. If the
            layers are supplied, the forward pass will use the more efficient forward
            Laplacian framework and return a list of dictionaries containing the push-
            forwards through all layers.
        X: Input for the interior loss. Has shape `(batch_size, 1 + dim_Omega)`. One
            datum `X[n]` has coordinates `(t, x_1, x_2 , ..., x_dim_Omega)`.
        y: Target for the interior loss. Has shape `(batch_size, 1)`.
        mu: Vector field. Maps an un-batched input `x` to a tensor `mu(x)` of shape
            `(dim_Omega)`.
        sigma: Diffusivity matrix. Maps `X` to a tensor `sigma(X)` of shape
            `(batch_size, dim_Omega, k)` with arbitrary `k` (usually `k = dim_Omega`).

    Returns:
        The differentiable interior loss, differentiable residual, and intermediates
        of the computation graph that can be used to compute (approximate) curvature.

    Raises:
        ValueError: If the model is not a PyTorch `Module` or a list of layers.
    """
    if isinstance(model, list) and all(isinstance(layer, Module) for layer in model):
        warn("Inefficient implementation!")
        model = Sequential(*model)
    elif not isinstance(model, Module):
        raise NotImplementedError

    # compute ∂p/∂t + div(p μ) = div[ p(t, x) (1, μ(t, x))ᵀ ]
    def p_times_mu(x: Tensor) -> Tensor:
        """Compute the product between p(t, x) and the augmented μ(t, x).

        Args:
            x: Un-batched input (time and spatial coordinates).

        Returns:
            Product of p(t, x) and augmented μ(t, x).
        """
        mu_x = mu(x)
        augment = ones(1, dtype=mu_x.dtype, device=mu_x.device)
        mu_x_augmented = cat([augment, mu_x])
        return model(x) * mu_x_augmented

    dp_dt_plus_div_p_times_mu = autograd_input_divergence(p_times_mu, X)

    # compute 1/2 Tr(σ σᵀ ∂²p/∂x²)
    hessian_X = autograd_input_hessian(model, X)  # [batch_size, d + 1, d + 1]
    hessian_spatial = hessian_X[:, 1:, 1:]  # [batch_size, d, d]

    sigma_X = sigma(X)
    sigma_outer = einsum(sigma_X, sigma_X, "batch i j, batch k j -> batch i k")
    sigma_outer_hessian = einsum(
        sigma_outer, hessian_spatial, "batch i k, batch k j -> batch i j"
    )
    tr_sigma_outer_hessian = einsum(sigma_outer_hessian, "batch i i -> batch")

    # compute residual and loss
    residual = (dp_dt_plus_div_p_times_mu - 0.5 * tr_sigma_outer_hessian) - y
    loss = 0.5 * (residual**2).mean()
    return loss, residual, None


def evaluate_boundary_loss(
    model: Union[Module, List[Module]], X: Tensor, y: Tensor
) -> Tuple[Tensor, Tensor, Union[List[Tensor], None]]:
    """Evaluate the boundary loss.

    Args:
        model: The model.
        X: Input for the boundary loss. Has shape `(batch_size, 1 + dim_Omega)`.
        y: Target for the boundary loss. Has shape `(batch_size, 1)`.

    Returns:
        The differentiable boundary loss, the differentiable residual, and a list of
        intermediates of the computation graph that can be used to compute (approximate)
        curvature.

    Raises:
        ValueError: If the model is not a Module or a list of Modules.
    """
    if isinstance(model, Module):
        output = model(X)
        intermediates = None
    elif isinstance(model, list) and all(isinstance(layer, Module) for layer in model):
        intermediates = manual_forward(model, X)
        output = intermediates[-1]
    else:
        raise ValueError(
            "Model must be a Module or a list of Modules that form a sequential model."
            f"Got: {model}."
        )
    residual = output - y
    return 0.5 * (residual**2).mean(), residual, intermediates


def mu_isotropic_gaussian(x: Tensor) -> Tensor:
    """Vector field for isotropic Gaussian case.

    Args:
        x: Un-batched input of shape `(1 + dim_Omega)` containing time and spatial
            coordinates.

    Returns:
        The vector field as tensor of shape `(dim_Omega)`.
    """
    return -0.5 * x[1:]


def sigma_isotropic_gaussian(X: Tensor) -> Tensor:
    """Diffusivity matrix for isotropic Gaussian case.

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


def p_isotropic_gaussian(X: Tensor) -> Tensor:
    """Isotropic Gaussian solution to the Fokker-Planck equation.

    Args:
        X: Batched quadrature points of shape `(N, d_Omega + 1)`.

    Returns:
        The function values as tensor of shape `(N, 1)`.
    """
    exp_t = (-X[:, 0]).exp()
    covariance = exp_t + 2 * (1 - exp_t)  # [batch_size]

    output = zeros_like(covariance)  # [batch_size]

    batch_size, d = X.shape
    d -= 1

    # TODO Implement more efficiently
    for n in range(batch_size):
        mean = zeros(d, device=X.device, dtype=X.dtype)
        cov = covariance[n] * eye(d, device=X.device, dtype=X.dtype)
        dist = MultivariateNormal(mean, cov)
        spatial_n = X[n, 1:]
        output[n] = dist.log_prob(spatial_n).exp()

    return output.unsqueeze(-1)
