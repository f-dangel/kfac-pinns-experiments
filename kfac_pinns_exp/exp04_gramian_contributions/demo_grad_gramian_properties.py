"""Numerically verify properties of the gradients used to form the Gramian."""

from itertools import product
from typing import List

from torch import Tensor, allclose, manual_seed, rand, zeros
from torch.nn import Linear, Module, Parameter, ReLU, Sequential, Sigmoid

from kfac_pinns_exp.exp04_gramian_contributions.demo_gramian_contributions import (
    CHILDREN,
    get_block_idx,
    get_layer_idx_and_name,
    gram_grads_term,
)
from kfac_pinns_exp.gramian_utils import autograd_gram_grads


def gram_grad_is_zero(
    layers: List[Module], X: Tensor, param: Parameter, child: str
) -> bool:
    """Check if the Gramian gradient contribution of a parameter is zero.

    Args:
        layers: List of layers in the model.
        X: Input data. First dimension is batch dimension.
        param: Parameter whose Gram gradient is checked.
        child: Child node whose contribution is checked.
            Possible values are `'output'`, `'grad_input'`, `'hess_input'`.

    Returns:
        `True` if the Gramian gradient contribution of the parameter is zero.
    """
    layer_idx, param_name = get_layer_idx_and_name(param, layers)
    contribution = gram_grads_term(layers, X, layer_idx, param_name, child)
    batch_size = X.shape[0]
    zero_contribution = zeros(
        batch_size, *param.shape, dtype=param.dtype, device=param.device
    )
    return allclose(contribution, zero_contribution)


def gram_grad_only_contributions(
    layers: List[Module], X: Tensor, param: Parameter, children: List[str]
) -> bool:
    """Check if Gram gradient contribution of a parameter is only from certain children.

    Args:
        layers: List of layers in the model.
        X: Input data. First dimension is batch dimension.
        param: Parameter whose Gram gradient is checked.
        children: List of children whose contributions are checked.
            Possible values are `'output'`, `'grad_input'`, `'hess_input'`.

    Returns:
        `True` if the Gramian gradient contribution of the parameter is only from the
        specified children.
    """
    model = Sequential(*layers)
    param_pos = get_block_idx(param, model)
    param_name = [name for name, _ in model.named_parameters()][param_pos]
    (truth,) = autograd_gram_grads(model, X, [param_name])

    layer_idx, param_name = get_layer_idx_and_name(param, layers)
    contribution = sum(
        gram_grads_term(layers, X, layer_idx, param_name, child) for child in children
    )
    return allclose(contribution, truth)


def main_sigmoid_net():
    """Check the bias contributions to the Gram gradients on an MLP with sigmoids.

    - Biases only contribute to the Laplacian through the forward pass.
      Hence, the terms stemming from the children `'grad_input'` and `'hess_input'`
      should be exactly zero.

    - Last layer weight does not contribute to the Hessian term. Hence, its
      terms stemming from the `'hess_input'` child should be exactly exactly zero.
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

    # check bias properties
    biases = [layer.bias for layer in layers if hasattr(layer, "bias")]

    # biases are not used in backward and Hessian backward procedure
    BIAS_UNUSED = ["grad_input", "hess_input"]
    for bias, child in product(biases, BIAS_UNUSED):
        assert gram_grad_is_zero(layers, X, bias, child)

    # biases are only used in the forward pass
    for bias in biases:
        assert gram_grad_only_contributions(layers, X, bias, ["output"])

    # check last layer weight properties
    assert isinstance(layers[-1], Linear)
    last_weight = layers[-1].weight
    # last weights are not used in the Hessian backward pass
    assert gram_grad_is_zero(layers, X, last_weight, "hess_input")

    # last weights are only used in the forward pass and the backward pass
    assert gram_grad_only_contributions(
        layers, X, last_weight, ["output", "grad_input"]
    )


def main_relu_net():
    """Check the bias contributions to the Gram gradients on an MLP with ReLUs.

    - The NN function is piecewise affine, hence the Laplacian is zero, and the
      Gram gradients are, too.
    """
    # setup
    manual_seed(4)
    batch_size = 100
    X = rand(batch_size, 5)
    layers = [
        Linear(5, 4),
        ReLU(),
        Linear(4, 3),
        ReLU(),
        Linear(3, 2),
        ReLU(),
        Linear(2, 1),
    ]

    # check any contribution (parameter and child) is exactly zero
    params = []
    for layer in layers:
        params.extend(iter(layer.parameters()))

    for param, child in product(params, CHILDREN):
        assert gram_grad_is_zero(layers, X, param, child)


if __name__ == "__main__":
    main_sigmoid_net()
    main_relu_net()
