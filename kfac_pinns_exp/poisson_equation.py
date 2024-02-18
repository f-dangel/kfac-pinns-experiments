"""Functionality for solving the Poisson equation."""

from einops import einsum
from torch import Tensor
from torch.nn import Module

from kfac_pinns_exp.autodiff_utils import autograd_gramian, autograd_input_hessian


def evaluate_interior_gramian(model: Module, X) -> Tensor:
    """Evaluate the interior loss' Gramian.

    Args:
        model: The model.
        X: Input for the interior loss.

    Returns:
        The interior loss Gramian.
    """
    batch_size = X.shape[0]
    param_names = [n for n, _ in model.named_parameters()]
    gramian = autograd_gramian(model, X, param_names, loss_type="interior")
    return gramian.div_(batch_size)


def evaluate_interior_loss(model: Module, X: Tensor, y: Tensor) -> Tensor:
    """Evaluate the interior loss.

    Args:
        model: The model.
        X: Input for the interior loss.
        y: Target for the interior loss.

    Returns:
        The differentiable interior loss.
    """
    input_hessian = autograd_input_hessian(model, X)
    laplacian = einsum(input_hessian, "batch i i -> batch i")
    return 0.5 * ((laplacian + y) ** 2).mean()


def evaluate_boundary_gramian(model: Module, X, approximation: str = "full") -> Tensor:
    """Evaluate the boundary loss' Gramian.

    Args:
        model: The model.
        X: Input for the boundary loss.
        approximation: The approximation to use for the Gramian. Default: `'full'`.
            Also supports `'diagonal'`.


    Returns:
        The boundary loss Gramian.
    """
    assert approximation in {"full", "diagonal"}
    batch_size = X.shape[0]
    param_names = [n for n, _ in model.named_parameters()]
    gramian = autograd_gramian(model, X, param_names, loss_type="boundary")
    return gramian.div_(batch_size)


def evaluate_boundary_loss(model: Module, X: Tensor, y: Tensor) -> Tensor:
    """Evaluate the boundary loss.

    Args:
        model: The model.
        X: Input for the boundary loss.
        y: Target for the boundary loss.

    Returns:
        The differentiable boundary loss.
    """
    output = model(X)
    return 0.5 * ((output - y) ** 2).mean()
