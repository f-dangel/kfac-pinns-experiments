"""Functionality to manually compute derivatives of neural networks."""

from typing import List, Optional

from einops import einsum
from torch import Tensor, ones_like, zeros_like
from torch.nn import Linear, Module, Sigmoid


def manual_forward(layers: List[Module], x: Tensor) -> List[Tensor]:
    """Apply a sequence of layers to an input.

    Args:
        layers: A list of layers.
        x: The input to the first layer. First dimension is batch dimension.

    Returns:
        A list of intermediate representations. First entry is the original input, last
        entry is the last layer's output.

    Raises:
        ValueError: If a layer uses in-place operations.
    """
    activations = [x]

    for layer in layers:
        if getattr(layer, "inplace", False):
            raise ValueError(
                f"Layers with in-place operations are not supported. Got {layer}."
            )

        x = layer(x)
        activations.append(x)

    return activations


def manual_backward(
    layers: List[Module],
    activations: List[Tensor],
    grad_output: Optional[Tensor] = None,
) -> List[Tensor]:
    """Backpropagate through all layers, computing gradients w.r.t. activations.

    Args:
        layers: A list of layers.
        activations: A list of intermediate representations from the forward pass.
            First entry is the original input, last entry is the last layer's output.
        grad_output: The vector that is backpropagated through the layers. Must have
            same size as the last element of `activations`. If not specified, a vector
            of ones will be used.

    Returns:
        A list of gradients. First entry is the gradient of the output w.r.t. the input,
        last entry is the gradient of the output w.r.t. the last layer's output, etc.

    Raises:
        ValueError: If the number of layers and activations do not match.
        ValueError: If `grad_output` is specified but has incorrect size.
    """
    if len(layers) != len(activations) - 1:
        raise ValueError(
            f"Number of layers ({len(layers)}) must equal number of activations - 1. "
            + f"got {len(activations)} activations."
        )

    grad_output = ones_like(activations[-1]) if grad_output is None else grad_output

    gradients = [zeros_like(a) for a in activations[:-1]] + [grad_output]

    # backpropagate, starting from the last layer
    for i in range(len(layers), 0, -1):
        layer = layers[i - 1]
        inputs, outputs = activations[i - 1], activations[i]
        grad_outputs = gradients[i]
        gradients[i - 1] += manual_backward_layer(layer, inputs, outputs, grad_outputs)

    return gradients


def manual_backward_layer(
    layer: Module, inputs: Tensor, outputs, grad_outputs: Tensor
) -> Tensor:
    """Backpropagate through a layer (output to input).

    Args:
        layer: The layer to backpropagate through.
        inputs: The input to the layer from the forward pass.
        outputs: The output of the layer from the forward pass.
        grad_outputs: The gradient of the loss w.r.t. the layer's output.

    Returns:
        The gradient of the loss w.r.t. the layer's input. Has same
        shape as the layer's input.

    Raises:
        NotImplementedError: If manual backpropagation for a layer is not implemented.
        RuntimeError: If `grad_outputs` or the return value have incorrect shapes.
    """
    if grad_outputs.shape != outputs.shape:
        raise RuntimeError(
            "Grad output must have same shape as output. "
            + f"Got {grad_outputs.shape} and {outputs.shape}."
        )

    if isinstance(layer, Linear):
        # ... denotes an arbitrary number of additional dimensions, e.g. sequence length
        grad_inputs = einsum(
            layer.weight, grad_outputs, "d_out d_in, batch ... d_out -> batch ... d_in"
        )
    elif isinstance(layer, Sigmoid):
        # σ' = σ(1 - σ)
        grad_inputs = grad_outputs * outputs * (1 - outputs)
    else:
        raise NotImplementedError(f"Backpropagation through {layer} not implemented.")

    if grad_inputs.shape != inputs.shape:
        raise RuntimeError(
            "Grad inputs must have same shape as inputs. "
            + f"Got {grad_inputs.shape} and {inputs.shape}."
        )

    return grad_inputs
