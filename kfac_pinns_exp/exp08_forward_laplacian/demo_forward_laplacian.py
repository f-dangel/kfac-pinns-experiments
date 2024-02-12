from typing import Dict, List

from einops import einsum
from torch import Tensor, allclose, eye, manual_seed, rand, zeros_like
from torch.nn import Linear, Module, Sequential, Sigmoid

from kfac_pinns_exp.autodiff_utils import autograd_input_hessian


def manual_forward_laplacian(
    layers: List[Module], x: Tensor
) -> List[Dict[str, Tensor]]:
    """Compute the NN prediction and Laplacian in one forward pass.

    Args:
        layers: A list of layers defining the NN.
        x: The input to the first layer. First dimension is batch dimension.

    Returns:
        A list of dictionaries, each containing the Taylor coefficients (0th-, 1st-, and
        summed 2nd-order) pushed through the layers layers.

    Raises:
        ValueError: If a layer uses in-place operations.
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

    # pass Taylor coefficients through the network
    result = [coefficients]

    for layer in layers:
        if getattr(layer, "inplace", False):
            raise ValueError(
                f"Layers with in-place operations are not supported. Got {layer}."
            )
        coefficients = manual_forward_laplacian_layer(layer, coefficients)
        result.append(coefficients)

    return result


def manual_forward_laplacian_layer(
    layer: Module, coefficients: Dict[str, Tensor]
) -> Dict[str, Tensor]:
    """Propagate the 0th, 1st, and summed 2nd-order Taylor coefficients through a layer.

    Args:
        layer: The layer to propagate the Taylor coefficients through.
        coefficients: A dictionary containing the Taylor coefficients.

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


def main():
    """Compute the forward Laplacian and compare with functorch on a simple example."""
    manual_seed(0)
    batch_size = 10
    X = rand(batch_size, 5)
    layers = [
        Linear(5, 3),
        Sigmoid(),
        Linear(3, 1),
        Sigmoid(),
    ]

    # automatic computation (via functorch)
    true_hessian_X = autograd_input_hessian(Sequential(*layers), X)
    true_laplacian_X = einsum(true_hessian_X, "batch d d -> ")

    # forward-Laplacian computation
    coefficients = manual_forward_laplacian(layers, X)
    laplacian_X = einsum(coefficients[-1]["laplacian"], "n d -> ")

    assert allclose(true_laplacian_X, laplacian_X)


if __name__ == "__main__":
    main()
