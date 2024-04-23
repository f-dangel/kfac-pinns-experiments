"""Manual implementation of Taylor-mode forward AD."""

from typing import Dict, List

from einops import einsum
from torch import Tensor, eye, gt, zeros
from torch.nn import Linear, Module, ReLU, Sigmoid, Tanh


def manual_forward_taylor(layers: List[Module], x: Tensor) -> List[Dict[str, Tensor]]:
    """Compute the NN's Taylor coefficients (up to second order) in one forward pass.

    Args:
        layers: A list of layers defining the NN.
        x: The input to the first layer. First dimension is batch dimension.

    Returns:
        A list of dictionaries, each containing the Taylor coefficients (0th-, 1st-, and
        2nd-order) pushed through the layers layers. Keys are `"c_0"`, `"c_1"`, and
        `"c_2"`.

    Raises:
        ValueError: If a layer uses in-place operations.
    """
    # inialize Taylor coefficients
    batch_size, feature_dims = x.shape[0], x.shape[1:]
    num_features = feature_dims.numel()
    gradients = (
        eye(num_features, dtype=x.dtype, device=x.device)
        .unsqueeze(0)
        .repeat(batch_size, 1, 1)
    ).reshape(
        batch_size,
        num_features,  # gradient elements
        *feature_dims,
    )
    hessians = zeros(
        batch_size,
        num_features,  # Hessian rows
        num_features,  # Hessian columns
        *feature_dims,
        dtype=x.dtype,
        device=x.device,
    )
    coefficients = {"c_0": x, "c_1": gradients, "c_2": hessians}

    # pass Taylor coefficients through the network
    result = [coefficients]

    for layer in layers:
        if getattr(layer, "inplace", False):
            raise ValueError(
                f"Layers with in-place operations are not supported. Got {layer}."
            )
        coefficients = manual_forward_taylor_layer(layer, coefficients)
        result.append(coefficients)

    return result


def manual_forward_taylor_layer(
    layer: Module, coefficients: Dict[str, Tensor]
) -> Dict[str, Tensor]:
    """Propagate the 0th, 1st, and 2nd-order Taylor coefficients through a layer.

    Args:
        layer: The layer to propagate the Taylor coefficients through.
        coefficients: A dictionary containing the Taylor coefficients.

    Returns:
        A dictionary containing the new Taylor coefficients.

    Raises:
        NotImplementedError: If the layer type is not supported.
    """
    old_c_0 = coefficients["c_0"]
    old_c_1 = coefficients["c_1"]
    old_c_2 = coefficients["c_2"]

    new_c_0 = layer(old_c_0)

    if isinstance(layer, Linear):
        W = layer.weight
        new_c_1 = einsum(W, old_c_1, "d_out d_in, n d0 ... d_in -> n d0 ... d_out")
        new_c_2 = einsum(W, old_c_2, "d_out d_in, ... d_in -> ... d_out")
    elif isinstance(layer, Sigmoid):
        jac = new_c_0 * (1 - new_c_0)
        hess = jac * (1 - 2 * new_c_0)
        new_c_1 = einsum(old_c_1, jac, "n d0 ..., n ... -> n d0 ...")
        new_c_2 = einsum(
            hess,
            old_c_1,
            old_c_1,
            "n ..., n d0_row ..., n d0_col ... -> n d0_row d0_col ...",
        ) + einsum(jac, old_c_2, "n ..., n d0_row d0_col ... -> n d0_row d0_col ...")
    elif isinstance(layer, ReLU):
        jac = gt(old_c_0, 0).to(old_c_0.dtype)
        new_c_1 = einsum(old_c_1, jac, "n d0 ..., n ... -> n d0 ...")
        new_c_2 = einsum(
            jac, old_c_2, "n ..., n d0_row d0_col ... -> n d0_row d0_col ..."
        )
    elif isinstance(layer, Tanh):
        jac = 1 - new_c_0**2
        hess = -2 * new_c_0 * jac
        new_c_1 = einsum(old_c_1, jac, "n d0 ..., n ... -> n d0 ...")
        new_c_2 = einsum(
            hess,
            old_c_1,
            old_c_1,
            "n ..., n d0_row ..., n d0_col ... -> n d0_row d0_col ...",
        ) + einsum(jac, old_c_2, "n ..., n d0_row d0_col ... -> n d0_row d0_col ...")
    else:
        raise NotImplementedError(f"Layer type not supported: {layer}.")

    return {"c_0": new_c_0, "c_1": new_c_1, "c_2": new_c_2}
