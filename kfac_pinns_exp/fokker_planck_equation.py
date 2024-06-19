"""Implements functionality to support solving the Fokker-Planck equation."""

from math import sqrt
from typing import Callable, Dict, List, Tuple, Union
from warnings import warn

from einops import einsum
from torch import Tensor, cat, eye, tensor, zeros, zeros_like
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import Module, Sequential

from kfac_pinns_exp.autodiff_utils import (
    autograd_input_divergence,
    autograd_input_hessian,
)


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
        X: Input for the interior loss.
        y: Target for the interior loss.
        mu: Vector field.
        sigma: Diffusivity matrix.

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

    # one datum X[n] has coordinates (t, x_1, x_2 , ..., x_d)

    # split time and spatial coordinates
    spatial = X[:, 1:].requires_grad_()  # [batch_size, d]
    t = X[:, [0]].requires_grad_()  # [batch_size, 1]
    X = cat((t, spatial), dim=1)  # [batch_size, d + 1]

    # compute ∂p/∂t + div(p μ) = div[ p(t, x) (1, μ(t, x))ᵀ ]
    def p_times_mu(x: Tensor) -> Tensor:
        """Compute the product between p(t, x) and the augmented μ(t, x).

        Args:
            x: Un-batched input (time and spatial coordinates).

        Returns:
            Product of p(t, x) and augmented μ(t, x).
        """
        mu_x = mu(x)
        mu_x_augmented = cat([tensor(1.0).expand_as(mu_x), mu_x], dim=0)
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
        X: Input for the boundary loss.
        y: Target for the boundary loss.

    Returns:
        The differentiable boundary loss, the differentiable residual, and a list of
        intermediates of the computation graph that can be used to compute (approximate)
        curvature.

    Raises:
        NotImplementedError: If the model is not a PyTorch `Module` or a list of layers.
    """
    if isinstance(model, list) and all(isinstance(layer, Module) for layer in model):
        warn("Inefficient implementation!")
        model = Sequential(*model)
    elif not isinstance(model, Module):
        raise NotImplementedError

    output = model(X)
    intermediates = None
    residual = output - y

    return 0.5 * (residual**2).mean(), residual, intermediates


def mu_isotropic_gaussian(x: Tensor) -> Tensor:
    return -0.5 * x[1:]


def sigma_isotropic_gaussian(X: Tensor) -> Tensor:
    (batch_size, dim) = X.shape
    dim_Omega = dim - 1
    # [batch_size, dim_Omega, dim_Omega]
    # [n, :, :] ∝ I
    return (sqrt(2) * eye(dim_Omega)).expand(batch_size, dim_Omega, dim_Omega)


def p_isotropic_gaussian(X: Tensor) -> Tensor:
    """Isotropic Gaussian solution to the Fokker-Planck equation.

    Args:
        X: Batched quadrature points of shape (N, d_Omega+1).

    Returns:
        The function values as tensor of shape (N, 1).
    """
    exp_t = (-X[:, 0]).exp()
    covariance = exp_t + 2 * (1 - exp_t)  # [batch_size]

    output = zeros_like(covariance)  # [batch_size]

    batch_size, d = X.shape
    d -= 1

    for n in range(batch_size):
        mean = zeros(d)
        cov = covariance[n] * eye(d)
        dist = MultivariateNormal(mean, cov)
        spatial_n = X[n, 1:]
        output[n] = dist.log_prob(spatial_n).exp()

    return output.unsqueeze(-1)
