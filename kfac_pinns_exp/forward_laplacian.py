"""Implementation of the forward Laplacian framework from Li et. al 2023."""

from typing import Dict, List, Optional, Union

from einops import einsum
from torch import Tensor, eye, gt, zeros_like
from torch.nn import Linear, Module, ReLU, Sigmoid, Tanh


def manual_forward_laplacian(
    layers: List[Module], x: Tensor, coordinates: Optional[List[int]] = None
) -> List[Dict[str, Tensor]]:
    """Compute the NN prediction and Laplacian in one forward pass.

    Args:
        layers: A list of layers defining the NN.
        x: The input to the first layer. First dimension is batch dimension.
        coordinates: List of indices specifying the Hessian diagonal entries
            that are summed into the Laplacian. If `None`, all diagonal entries
            are summed. Default: `None`.

    Returns:
        A list of dictionaries, each containing the Taylor coefficients (0th-, 1st-, and
        summed 2nd-order) pushed through the layers layers.

    Raises:
        ValueError: If a layer uses in-place operations.
        ValueError: If coordinates are not unique or out of range.
    """
    # inialize Taylor coefficients
    batch_size, feature_dims = x.shape[0], x.shape[1:]
    num_features = feature_dims.numel()
    directional_grad_init = (
        eye(num_features, dtype=x.dtype, device=x.device)
        .unsqueeze(0)
        .repeat(batch_size, 1, 1)
    ).reshape(batch_size, num_features, *feature_dims)
    laplacian_init = zeros_like(x)
    coefficients = {
        "forward": x,
        "directional_gradients": directional_grad_init,
        "laplacian": laplacian_init,
    }

    if coordinates is not None:
        if len(set(coordinates)) != len(coordinates) or len(coordinates) == 0:
            raise ValueError(
                f"Coordinates must be unique and non-empty. Got {coordinates}."
            )
        if any(c < 0 or c >= num_features for c in coordinates):
            raise ValueError(
                f"Coordinates must be in the range [0, {num_features})."
                f" Got {coordinates}."
            )

    # pass Taylor coefficients through the network
    result = [coefficients]

    for layer in layers:
        if getattr(layer, "inplace", False):
            raise ValueError(
                f"Layers with in-place operations are not supported. Got {layer}."
            )
        coefficients = manual_forward_laplacian_layer(
            layer, coefficients, coordinates=coordinates
        )
        result.append(coefficients)

    return result


def manual_forward_laplacian_layer(
    layer: Module, coefficients: Dict[str, Tensor], coordinates: Union[List[int], None]
) -> Dict[str, Tensor]:
    """Propagate the 0th, 1st, and summed 2nd-order Taylor coefficients through a layer.

    Args:
        layer: The layer to propagate the Taylor coefficients through.
        coefficients: A dictionary containing the Taylor coefficients.
        coordinates: List of indices specifying the Hessian diagonal entries
            that are summed into the Laplacian. If `None`, all diagonal entries
            are summed.

    Returns:
        A dictionary containing the new Taylor coefficients.

    Raises:
        NotImplementedError: If the layer type is not supported.
    """
    old_forward = coefficients["forward"]
    old_directional_gradients = coefficients["directional_gradients"]
    old_laplacian = coefficients["laplacian"]

    new_forward = layer(old_forward)

    if isinstance(layer, Linear):
        new_directional_gradients = einsum(
            layer.weight,
            old_directional_gradients,
            "d_out d_in, n d0 ... d_in -> n d0 ... d_out",
        )
        new_laplacian = einsum(
            layer.weight, old_laplacian, "d_out d_in, ... d_in -> ... d_out"
        )
    elif isinstance(layer, Sigmoid):
        jac = new_forward * (1 - new_forward)
        hess = jac * (1 - 2 * new_forward)
        new_directional_gradients = einsum(
            old_directional_gradients, jac, "n d0 ..., n ... -> n d0 ..."
        )
        # only use the relevant coordinates for the Laplacian
        if coordinates is not None:
            old_directional_gradients = old_directional_gradients[:, coordinates]
        new_laplacian = einsum(
            hess, old_directional_gradients**2, "n ..., n d0 ... -> n ..."
        ) + einsum(jac, old_laplacian, "n ..., n ... -> n ... ")
    elif isinstance(layer, ReLU):
        jac = gt(old_forward, 0).to(old_forward.dtype)
        new_directional_gradients = einsum(
            old_directional_gradients, jac, "n d0 ..., n ... -> n d0 ..."
        )
        new_laplacian = einsum(jac, old_laplacian, "n ..., n ... -> n ... ")
    elif isinstance(layer, Tanh):
        jac = 1 - new_forward**2
        hess = -2 * new_forward * jac
        new_directional_gradients = einsum(
            old_directional_gradients, jac, "n d0 ..., n ... -> n d0 ..."
        )
        # only use the relevant coordinates for the Laplacian
        if coordinates is not None:
            old_directional_gradients = old_directional_gradients[:, coordinates]
        new_laplacian = einsum(
            hess, old_directional_gradients**2, "n ..., n d0 ... -> n ..."
        ) + einsum(jac, old_laplacian, "n ..., n ... -> n ... ")
    else:
        raise NotImplementedError(f"Layer type not supported: {layer}.")

    return {
        "forward": new_forward,
        "directional_gradients": new_directional_gradients,
        "laplacian": new_laplacian,
    }
