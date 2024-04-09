"""Utility functions for KFAC."""

from typing import Dict, List, Tuple

from torch import Tensor, arange, cat, eye, zeros
from torch.nn import Linear, Module


def check_layers_and_initialize_kfac(
    layers: List[Module], initialize_to_identity: bool = False
) -> Dict[int, Tuple[Tensor, Tensor]]:
    """Verify all layers are supported and initialize the KFAC factors.

    Args:
        layers: The list of layers in the neural network.
        initialize_to_identity: Whether to initialize the KFAC factors to the identity
            matrix. If `False`, the factors are initialized to zero. Default is `False`.

    Returns:
        A dictionary whose keys are the layer indices and whose values are the two
        Kronecker factors.

    Raises:
        NotImplementedError: If a layer with parameters is not a linear layer with bias
            and both parameters differentiable.
    """
    kfacs = {}

    for layer_idx, layer in enumerate(layers):
        if list(layer.parameters()) and not isinstance(layer, Linear):
            raise NotImplementedError("Only parameters in linear layers are supported.")
        if isinstance(layer, Linear):
            if layer.bias is None:
                raise NotImplementedError("Only layers with bias are supported.")
            if any(not p.requires_grad for p in layer.parameters()):
                raise NotImplementedError("All parameters must require gradients.")
            weight = layer.weight
            kwargs = {"dtype": weight.dtype, "device": weight.device}
            d_out, d_in = weight.shape
            if initialize_to_identity:
                A = eye(d_in + 1, **kwargs)
                B = eye(d_out, **kwargs)
            else:
                A = zeros(d_in + 1, d_in + 1, **kwargs)
                B = zeros(d_out, d_out, **kwargs)
            kfacs[layer_idx] = (A, B)

    return kfacs


def gramian_basis_to_kfac_basis(gramian: Tensor, dim_A: int, dim_B: int) -> Tensor:
    """Rearrange the Gramian such that its basis matches that of KFAC.

    For a linear layer with weight `W` and bias `b`, the Gramian's basis is
    `(W.flatten().T, b.T).T` while KFAC's basis is `(W, b).flatten()` which is
    different.

    Args:
        gramian: The Gramian matrix.
        dim_A: The dimension of the first (input-based) Kronecker factor.
        dim_B: The dimension of the second (grad-output-based) Kronecker factor.

    Returns:
        The rearranged Gramian.
    """
    # create a 1d array which maps current positions to new positions via slicing,
    # i.e. its i-th entry contains the position of the element in the old basis
    # which should be the i-th vector in the new basis
    rearrange = cat(
        [
            arange(dim_B * dim_A).reshape(dim_B, dim_A),
            arange(dim_B * dim_A, dim_B * (dim_A + 1)).unsqueeze(1),
        ],
        dim=1,
    ).flatten()
    return gramian[rearrange, :][:, rearrange]
