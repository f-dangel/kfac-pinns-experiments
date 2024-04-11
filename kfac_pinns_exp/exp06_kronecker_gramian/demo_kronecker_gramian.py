"""Compute the Kronecker approximation of the Laplacian's Gramian."""

from functools import partial
from typing import List, Tuple

from einops import einsum
from torch import Tensor, autograd, manual_seed, rand, zeros
from torch.nn import Linear, Module, Sequential, Sigmoid
from torch.utils.hooks import RemovableHandle

from kfac_pinns_exp.exp04_gramian_contributions.demo_gramian_contributions import (
    CHILDREN,
)
from kfac_pinns_exp.hooks_gram_kfac_linear import (  # from_grad_input,; from_hess_input,
    from_output,
)
from kfac_pinns_exp.manual_differentiation import (
    manual_backward,
    manual_forward,
    manual_hessian_backward,
)


def compute_kfag(
    layers: List[Module],
    X: Tensor,
    children: Tuple[str, ...] = ("output", "grad_input", "hess_input"),
) -> List[List[Tensor]]:
    """Compute the Kronecker approximation of the Laplacian's Gramian.

    Args:
        layers: A list of layers defining a sequential neural network.
        X: The input data. First dimension is the batch dimension.
        children: A tuple of strings specifying which children in the compute
            should be considered for the Kronecker approximation.
            Valid entries are: `'output'`, `'grad_input'`, `'hess_input'`.
            Default: `('output', 'grad_input', 'hess_input')`.

    Returns:
        A list of tuples `(A, B)` where `A` and `B` are the Kronecker factors of the
        approximation for the Laplacian's Gramian `G ≈ A ⊗ B` (in column-flattening
        convention) for each layer with parameters.

    Raises:
        ValueError: If `children` contains an invalid value.
    """
    if any(c not in CHILDREN for c in children):
        raise ValueError(
            f"Invalid children: {children}. Valid children are: {CHILDREN}."
        )

    activations = manual_forward(layers, X)
    gradients = manual_backward(layers, activations)
    hessians = manual_hessian_backward(layers, activations, gradients)
    laplacian = einsum(hessians[0], "batch d d ->")

    layer_idxs = [idx for idx, layer in enumerate(layers) if isinstance(layer, Linear)]
    has_bias = [layers[idx].bias is not None for idx in layer_idxs]
    hook_handles: List[RemovableHandle] = []

    # initialize Kronecker factors
    kronecker_factors = []
    for layer_idx, bias in zip(layer_idxs, has_bias):
        layer = layers[layer_idx]
        weight = layer.weight
        d_out, d_in = weight.shape
        if bias:
            d_in += 1

        # input based
        kfag_A = zeros(d_in, d_in, device=weight.device, dtype=weight.dtype)
        # incoming gradient based
        kfag_B = zeros(d_out, d_out, device=weight.device, dtype=weight.dtype)

        kronecker_factors.append([kfag_A, kfag_B])

    # install hooks
    for layer_idx, bias, (kfag_A, kfag_B) in zip(
        layer_idxs, has_bias, kronecker_factors
    ):
        # extract quantities required for the Gramian of a layer's parameter
        layer_input = activations[layer_idx]
        layer_output = activations[layer_idx + 1]
        layer_grad_input = gradients[layer_idx]
        layer_grad_output = gradients[layer_idx + 1]
        layer_hess_input = hessians[layer_idx]
        layer_hess_output = hessians[layer_idx + 1]

        if "output" in children:
            handle = layer_output.register_hook(
                partial(
                    from_output,
                    layer_input=layer_input,
                    bias=bias,
                    kfag_A_accumulator=kfag_A,
                    kfag_B_accumulator=kfag_B,
                )
            )
            hook_handles.append(handle)
        if "grad_input" in children:
            raise NotImplementedError
            # layer_grad_input.register_hook(
            #     partial(
            #         from_grad_input,
            #         layer_input=layer_input,
            #         param_name=param_name,
            #         accumulator=gram_grads,
            #     )
            # )
        if "hess_input" in children:
            raise NotImplementedError
            # handle = layer_hess_input.register_hook(
            #     partial(
            #         from_hess_input,
            #         layer_hess_output=layer_hess_output,
            #         weight=param,
            #         accumulator=gram_grads,
            #     )
            # )
            # hook_handles.append(handle)

    # compute the Gramian's Kronecker approximation, use `grad` to avoid writes
    # to `param.grad`
    params = [p for layer in layers for p in layer.parameters()]
    autograd.grad(laplacian, params, allow_unused=True)

    # remove hooks
    for handle in hook_handles:
        handle.remove()

    return kronecker_factors


if __name__ == "__main__":
    # setup
    manual_seed(0)
    batch_size = 10
    X = rand(batch_size, 5)
    layers = [
        Linear(5, 4),
        Sigmoid(),
        Linear(4, 3),
        Sigmoid(),
        Linear(3, 2),
        Sigmoid(),
        Linear(2, 1),
    ]
    model = Sequential(*layers)

    kfag = compute_kfag(layers, X, children=("output",))

    for A, B in kfag:
        print(A.shape, B.shape)
