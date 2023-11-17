"""Functionality to manually compute derivatives of neural networks."""

from typing import List

from torch import Tensor
from torch.nn import Module


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
