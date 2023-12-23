"""Illustrates computing per-node and per-parameter Gramian terms."""

from functools import partial
from itertools import product
from typing import List, Tuple

from einops import einsum, rearrange
from torch import Tensor, allclose, manual_seed, rand, zeros, zeros_like
from torch.nn import Linear, Module, Parameter, Sequential, Sigmoid

from kfac_pinns_exp.autodiff_utils import autograd_gramian
from kfac_pinns_exp.hooks_gram_grads_linear import (
    from_grad_input,
    from_hess_input,
    from_output,
)
from kfac_pinns_exp.manual_differentiation import (
    manual_backward,
    manual_forward,
    manual_hessian_backward,
)
from kfac_pinns_exp.utils import separate_into_tiles

CHILDREN = ("output", "grad_input", "hess_input")


def gram_grads_term(
    layers: List[Module], X: Tensor, layer_idx: int, param_name: str, child_name: str
) -> Tensor:
    """Compute the gradients that contribute to the Gramian caused by one child node.

    Args:
        layers: List of layers in the model.
        X: Input data. First dimension is batch dimension.
        layer_idx: Index of the layer whose gradient contribution is computed.
        param_name: Name of the parameter we want to use to compute the Gramian.
        child_name: Name of the child node whose gradient contribution is computed.
            Possible values are `'output'`, `'grad_input'`, `'hess_input'`.

    Returns:
        Tensor of shape `(batch_size, *p.shape)` where `p` is the parameter.
        The tensor contains the parameters gradient that contributes to the Gramian
        through the specified child.

    Raises:
        ValueError: If `child_name` is not one of the allowed values.
    """
    if child_name not in CHILDREN:
        raise ValueError(f"Unknown child name {child_name!r}. Supported: {CHILDREN}")

    activations = manual_forward(layers, X)
    gradients = manual_backward(layers, activations)
    hessians = manual_hessian_backward(layers, activations, gradients)
    laplacian = einsum(hessians[0], "batch d d ->")

    # extract quantities required for the Gramian of a layer's parameter
    layer_input = activations[layer_idx]
    layer_output = activations[layer_idx + 1]
    layer_grad_input = gradients[layer_idx]
    layer_grad_output = gradients[layer_idx + 1]
    layer_hess_input = hessians[layer_idx]
    layer_hess_output = hessians[layer_idx + 1]

    # install hooks that accumulate the gradients for the Gramian in `gram_grads`
    # { ∂(Δu(xᵢ)) / ∂W | i = 1, ..., batch_size }
    assert isinstance(layers[layer_idx], Linear)
    param = getattr(layers[layer_idx], param_name)
    gram_grads = zeros(X.shape[0], *param.shape, device=param.device, dtype=param.dtype)

    if param_name in {"bias", "weight"} and child_name == "output":
        layer_output.register_hook(
            partial(
                from_output,
                layer_input=layer_input,
                param_name=param_name,
                accumulator=gram_grads,
            )
        )
    if param_name == "weight" and child_name == "grad_input":
        layer_grad_input.register_hook(
            partial(
                from_grad_input,
                layer_grad_output=layer_grad_output,
                accumulator=gram_grads,
            )
        )
    if param_name == "weight" and child_name == "hess_input":
        layer_hess_input.register_hook(
            partial(
                from_hess_input,
                layer_hess_output=layer_hess_output,
                weight=param,
                accumulator=gram_grads,
            )
        )

    # backpropagate
    laplacian.backward()

    return gram_grads


def gramian_term(
    layers: List[Module],
    X: Tensor,
    layer1_idx: int,
    param1_name: str,
    param1_child: str,
    layer2_idx: int,
    param2_name: str,
    param2_child: str,
    flat_params: bool = False,
) -> Tensor:
    """Compute the contribution to the Gramian from two parameters and children.

    Computes the following:

    - `∇₁`, the contribution of the first specified child to `∇_{p₁} Δu` where `p₁` is
      the first specified parameter, and
    - `∇₂`, the contribution of the second specified child to `∇_{p₂} Δu` where `p₂`
      is the second specified parameter
    - returns the batch-sum of `∇₁ ∇₂ᵀ`

    Args:
        layers: List of layers in the model.
        X: Input data. First dimension is batch dimension.
        layer1_idx: Index of the first layer Gram gradient.
        param1_name: Name of the first layer Gram gradient parameter.
        param1_child: Name of the first layer Gram gradient child.
            Must be one of `'output'`, `'grad_input'`, `'hess_input'`.
        layer2_idx: Index of the second layer Gram gradient.
        param2_name: Name of the second layer Gram gradient parameter.
        param2_child: Name of the second layer Gram gradient child.
            Must be one of `'output'`, `'grad_input'`, `'hess_input'`.
        flat_params: If `True`, the Gramian is computed as a matrix product of the
            flattened gradients. Otherwise, parameter dimensions are preserved.

    Returns:
        Tensor of shape `(*p₁.shape, *p₂.shape)` or `(p1.numel(), p2.numel())` where
        `p₁` and `p₂` are the specified parameters.
    """
    # compute the gradients for the Gramian
    gram_grads1 = gram_grads_term(layers, X, layer1_idx, param1_name, param1_child)
    gram_grads2 = gram_grads_term(layers, X, layer2_idx, param2_name, param2_child)

    # form the Gramian contribution
    p1_axes = " ".join([f"p1_{i}" for i in range(gram_grads1.ndim - 1)])
    p2_axes = " ".join([f"p2_{i}" for i in range(gram_grads2.ndim - 1)])
    gram = einsum(
        gram_grads1,
        gram_grads2,
        f"batch {p1_axes}, batch {p2_axes} -> {p1_axes} {p2_axes}",
    )
    if flat_params:
        gram = rearrange(gram, f"{p1_axes} {p2_axes} -> ({p1_axes}) ({p2_axes})")

    return gram


def get_layer_idx_and_name(param: Parameter, layers: List[Module]) -> Tuple[int, str]:
    """Get the layer index and parameter name of a parameter.

    Args:
        param: Parameter to get the layer index and parameter name of.
        layers: List of layers to search for the parameter.

    Returns:
        Index of the layer containing the parameter and the name of the parameter.

    Raises:
        ValueError: If the parameter is not found in the list of layers.
    """
    for idx, layer in enumerate(layers):
        for name, layer_param in layer.named_parameters():
            if param.data_ptr() == layer_param.data_ptr():
                return idx, name

    raise ValueError(f"Parameter {param} not found in layers {layers}.")


def get_block_idx(param, model: Module) -> int:
    """Get the block index of a parameter.

    Args:
        param: Parameter to get the block index of.
        model: Model containing the parameter.

    Returns:
        Index of the parameter block in the sequential model.
    """
    block_ids = [p.data_ptr() for p in model.parameters()]
    return block_ids.index(param.data_ptr())


def main():
    """Compute the different contributions to the Gramian and check their sum.

    Checks that for all Gramian blocks, summing over all the contributions gives the
    full Gramian.
    """
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

    # compute the full Gramian with functorch and chunk it into blocks
    dims = [p.numel() for p in model.parameters()]
    blocked_gram = separate_into_tiles(
        autograd_gramian(model, X, [name for name, _ in model.named_parameters()]), dims
    )

    for param1, param2 in product(model.parameters(), model.parameters()):
        layer_idx1, param_name1 = get_layer_idx_and_name(param1, layers)
        layer_idx2, param_name2 = get_layer_idx_and_name(param2, layers)
        print(f"{layer_idx1} {param_name1} @ {layer_idx2} {param_name2}")

        # ground truth
        block_idx1 = get_block_idx(param1, model)
        block_idx2 = get_block_idx(param2, model)
        true_block = blocked_gram[block_idx1][block_idx2]

        # the sum over all contributions must match the true block
        manual_block = zeros_like(true_block)

        for child1, child2 in product(CHILDREN, CHILDREN):
            manual_block.add_(
                gramian_term(
                    layers,
                    X,
                    layer_idx1,
                    param_name1,
                    child1,
                    layer_idx2,
                    param_name2,
                    child2,
                    flat_params=True,
                )
            )

        same = allclose(manual_block, true_block)
        print(f"\tSame(auto, hook)? {same}")
        assert same


if __name__ == "__main__":
    main()
