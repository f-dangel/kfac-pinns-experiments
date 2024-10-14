"""PINN-related utility functions."""

from typing import Callable, Dict, List, Tuple, Union

from torch import Tensor, ones_like
from torch.autograd import grad
from torch.nn import Linear, Module

from kfac_pinns_exp.kfac_utils import compute_kronecker_factors
from kfac_pinns_exp.manual_differentiation import manual_forward
from kfac_pinns_exp.utils import bias_augmentation


def l2_error(model: Module, X: Tensor, u: Callable[[Tensor], Tensor]) -> Tensor:
    """Computes the L2 norm of the error = model - u on the domain Omega.

    Args:
        model: The model.
        X: randomly drawn points in Omega.
        u: Function to evaluate the manufactured solution.

    Returns:
        The L2 norm of the error.
    """
    y = (model(X) - u(X)) ** 2
    return y.mean() ** (1 / 2)


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


def evaluate_boundary_loss_and_kfac(
    layers: List[Module],
    X: Tensor,
    y: Tensor,
    ggn_type: str = "type-2",
    kfac_approx: str = "expand",
) -> Tuple[Tensor, Dict[int, Tuple[Tensor, Tensor]]]:
    """Evaluate the boundary loss and compute its KFAC approximation.

    Args:
        layers: The list of layers in the neural network.
        X: The input data.
        y: The target data.
        ggn_type: The type of GGN to compute. Can be `'empirical'`, `'type-2'`,
            or `'forward-only'`. Default: `'type-2'`.
        kfac_approx: The type of KFAC approximation to use. Can be `'expand'` or
            `'reduce'`. Default: `'expand'`.

    Returns:
        The (differentiable) boundary loss and a dictionary whose keys are the layer
        indices and whose values are the two Kronecker factors.
    """
    # Compute the NN prediction, boundary loss, and all intermediates
    loss, _, layer_inputs, layer_grad_outputs = (
        evaluate_boundary_loss_with_layer_inputs_and_grad_outputs(
            layers, X, y, ggn_type
        )
    )
    kfacs = compute_kronecker_factors(
        layers, layer_inputs, layer_grad_outputs, ggn_type, kfac_approx
    )
    return loss, kfacs


def evaluate_boundary_loss_with_layer_inputs_and_grad_outputs(
    layers: List[Module], X: Tensor, y: Tensor, ggn_type: str
) -> Tuple[Tensor, Tensor, Dict[int, Tensor], Dict[int, Tensor]]:
    """Compute the boundary loss, residual & inputs+output gradients of Linear layers.

    Args:
        layers: The list of layers that form the neural network.
        X: The input data.
        y: The target data.
        ggn_type: The type of GGN to use. Can be `'type-2'`, `'empirical'`, or
            `'forward-only'`.

    Returns:
        A tuple containing the loss, residual, inputs of the Linear layers, and the out-
        put gradients of the Linear layers. The layer inputs are augmented with ones to
        account for the bias term.
    """
    layer_idxs = [
        idx
        for idx, layer in enumerate(layers)
        if (
            isinstance(layer, Linear)
            and layer.bias is not None
            and layer.bias.requires_grad
            and layer.weight.requires_grad
        )
    ]
    loss, residual, intermediates = evaluate_boundary_loss(layers, X, y)

    # collect all layer inputs
    layer_inputs = {
        idx: bias_augmentation(intermediates[idx].detach(), 1) for idx in layer_idxs
    }

    if ggn_type == "forward-only":
        return loss, layer_inputs, {}

    # collect all layer output gradients
    layer_outputs = [intermediates[idx + 1] for idx in layer_idxs]
    error = get_backpropagated_error(residual, ggn_type)
    grad_outputs = grad(residual, layer_outputs, grad_outputs=error, retain_graph=True)
    layer_grad_outputs = {idx: g for g, idx in zip(grad_outputs, layer_idxs)}

    return loss, residual, layer_inputs, layer_grad_outputs


def get_backpropagated_error(residual: Tensor, ggn_type: str) -> Tensor:
    """Get the error which is backpropagated to compute the second KFAC factor.

    Args:
        residual: The residual tensor which is squared then averaged to compute
            the loss.
        ggn_type: The type of GGN approximation. Can be "type-2" or "empirical".

    Returns:
        The error tensor. Has same shape as `residual`.

    Raises:
        NotImplementedError: If the `ggn_type` is not supported.
    """
    batch_size = residual.shape[0]
    if ggn_type == "type-2":
        return ones_like(residual) / batch_size
    elif ggn_type == "empirical":
        return residual.clone().detach() / batch_size
    raise NotImplementedError(f"GGN type {ggn_type} is not supported.")
