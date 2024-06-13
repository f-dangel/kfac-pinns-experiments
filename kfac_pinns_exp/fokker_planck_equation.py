"""Implements functionality to support solving the Fokker-Planck equation."""

from typing import Callable, Dict, List, Tuple, Union

from torch import Tensor, cat, tensor, zeros
from torch.autograd import grad
from torch.nn import Module

from kfac_pinns_exp.autodiff_utils import (
    autograd_input_hessian,
    autograd_input_jacobian,
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
        ValueError: If the model is not a Module or a list of Modules.
    """
    # use autograd to compute the Laplacian
    if isinstance(model, Module):
        # X[0] = (t, x_1, x_2 , ..., x_d)
        intermediates = None

        hessian_p = autograd_input_hessian(model, X)
        jacobian_p = autograd_input_jacobian(model, X)
        jacobian_p_t = jacobian_p[:, 0]
        # jacobian_p_x = jacobian_p[:, 1:]

        # [batch_size, d]
        X_spatial = X[:, 1:].requires_grad_()
        # [batch_size, 1]
        t = X[:, [0]].requires_grad_()

        # [batch_size, d + 1]
        X = cat((t, X_spatial), dim=1)

        # [batch_size, 1]
        p = model(X)

        # [batch_size, d]
        mu_t = mu(t)  # mu_t[0] = (mu_1, ..., mu_d)
        p_mu = p * mu_t

        batch_size, D = p_mu.shape

        div = zeros(batch_size, 1)

        for n in range(batch_size):
            for d in range(D):
                # [batch_size, d]
                # ∂p_mu[n, d] / ∂x_d
                (g,) = grad(p_mu[n, d], X_spatial, retain_graph=True)
                div[n].add_(g[n, d])

        sigma_t = sigma(t)

        # t.requires_grad_()

        # batch_size, d = t.shape[1]
        # for n in range(batch_size):
        #     jacobian_mu_n_t =  grad(mu_t, t)

        # laplacian = einsum(input_hessian, "batch i i -> batch").unsqueeze(-1)
    else:
        raise NotImplementedError
    # # use the forward Laplacian framework
    # elif isinstance(model, list) and all(isinstance(layer, Module) for layer in model):
    #     intermediates = manual_forward_laplacian(model, X)
    #     laplacian = intermediates[-1]["laplacian"]
    # else:
    #     raise ValueError(
    #         "Model must be a Module or a list of Modules that form a sequential model."
    #         f"Got: {model}."
    #     )
    # residual = laplacian + y
    # return 0.5 * (residual**2).mean(), residual, intermediates


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
        ValueError: If the model is not a Module or a list of Modules.
    """
    raise NotImplementedError
    # if isinstance(model, Module):
    #     output = model(X)
    #     intermediates = None
    # elif isinstance(model, list) and all(isinstance(layer, Module) for layer in model):
    #     intermediates = manual_forward(model, X)
    #     output = intermediates[-1]
    # else:
    #     raise ValueError(
    #         "Model must be a Module or a list of Modules that form a sequential model."
    #         f"Got: {model}."
    #     )
    # residual = output - y
    # return 0.5 * (residual**2).mean(), residual, intermediates
