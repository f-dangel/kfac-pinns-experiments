"""Utility functions for KFAC."""

from typing import Dict, List, Tuple

from einops import rearrange, reduce
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


def compute_kronecker_factors(
    layers: List[Module],
    inputs: Dict[int, Tensor],
    grad_outputs: Dict[int, Tensor],
    ggn_type: str,
    kfac_approx: str,
) -> Dict[int, Tuple[Tensor, Tensor]]:
    """Compute KFAC's Kronecker factors from layers inputs and output gradients.

    Args:
        layers: The list of layers in the neural network.
        inputs: A dictionary whose keys are the indices of layers for which the
            Kronecker factors are computed. The value is the input to the layer from
            the forward pass, arranged into a matrix.
        grad_outputs: A dictionary from layer indices to gradient of the loss with
            respect to their output. Can be empty if the GGN type is `'forward-only`.
        ggn_type: The type of GGN to use. Can be `'forward-only'`, `'type-2'`, or
            `'empirical'`.
        kfac_approx: The type of KFAC approximation to use. Can be `'expand'` or
            `'reduce'`.

    Raises:
        ValueError: If `kfac_approx` is not `'expand'` or `'reduce'`.
        ValueError: If `ggn_type` is not `'forward-only'`, `'type-2'`, or `'empirical'`.

    Returns:
        A dictionary whose keys are the layer indices and whose values are the two
        Kronecker factors.
    """
    if kfac_approx not in {"expand", "reduce"}:
        raise ValueError(
            f"kfac_approx must be 'expand' or 'reduce'. Got {kfac_approx}."
        )
    if ggn_type not in {"forward-only", "type-2", "empirical"}:
        raise ValueError(
            "ggn_type must be 'forward-only', 'type-2', or 'empirical'."
            f" Got {ggn_type}."
        )

    kfacs = check_layers_and_initialize_kfac(layers, initialize_to_identity=False)
    (batch_size,) = {Z.shape[0] for Z in inputs.values()}

    # Compute input-based Kronecker factors
    for layer_idx, (A, _) in kfacs.items():
        Z = inputs.pop(layer_idx)
        if kfac_approx == "expand":
            Z = rearrange(Z, "batch ... d_in -> (batch ...) d_in")
        else:  # KFAC-reduce
            Z = reduce(Z, "batch ... d_in -> batch d_in", "mean")
        A.add_(Z.T @ Z, alpha=1 / Z.shape[0])

    # Compute output-gradient-based Kronecker factor
    for layer_idx, (_, B) in kfacs.items():
        if ggn_type == "forward-only":
            # set all grad-output Kronecker factors to identity
            B.fill_diagonal_(1.0)
        else:
            G = grad_outputs.pop(layer_idx)
            if kfac_approx == "expand":
                G = rearrange(G, "batch ... d_out -> (batch ...) d_out")
            else:  # KFAC-reduce
                G = reduce(G, "batch ... d_out -> batch d_out", "sum")
            B.add_(G.T @ G, alpha=batch_size)

    return kfacs


def gramian_basis_to_kfac_basis(mat_or_vec: Tensor, dim_A: int, dim_B: int) -> Tensor:
    """Rearrange the Gramian such that its basis matches that of KFAC.

    For a linear layer with weight `W` and bias `b`, the Gramian's basis is
    `(W.flatten().T, b.T).T` while KFAC's basis is `(W, b).flatten()` which is
    different.

    Args:
        mat_or_vec: A matrix or vector in the Gramian's basis.
        dim_A: The dimension of the first (input-based) Kronecker factor.
        dim_B: The dimension of the second (grad-output-based) Kronecker factor.

    Returns:
        The rearranged matrix or vector in the KFAC basis.

    Raises:
        ValueError: If the supplied tensor is not 1d or 2d.
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
    if mat_or_vec.ndim == 2:
        return mat_or_vec[rearrange, :][:, rearrange]
    elif mat_or_vec.ndim == 1:
        return mat_or_vec[rearrange]
    else:
        raise ValueError(f"Only 1,2d tensors are supported. Got {mat_or_vec.ndim}d.")
