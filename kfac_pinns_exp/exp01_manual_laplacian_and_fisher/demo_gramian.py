"""Demonstrate Gramian computation on a small toy MLP."""

from typing import List

from einops import einsum
from torch import Tensor, allclose, autograd, manual_seed, rand, zeros, zeros_like
from torch.nn import Linear, Module, Sequential, Sigmoid

from kfac_pinns_exp.autodiff_utils import autograd_gramian
from kfac_pinns_exp.manual_differentiation import (
    manual_backward,
    manual_forward,
    manual_hessian_backward,
)


def manual_laplace_autograd_gramian(
    layers: List[Module], X: Tensor, layer_idx: int, param_name: str
) -> Tensor:
    """Compute the Gramian of the Laplacian of a parameter.

    The Laplacian is computed manually, the gradients for the Gramian are
    computed via autograd.

    Args:
        layers: List of layers in the model.
        X: Input data. First dimension is batch dimension.
        layer_idx: Index of the layer that contains the parameter we want to use to
            compute the Gramian.
        param_name: Name of the parameter we want to use to compute the Gramian.

    Returns:
        Gramian of the neural network's Laplacian w.r.t. the parameter. Has shape
        `[*p.shape, *p.shape]` where `p` denotes the parameter.
    """
    param = getattr(layers[layer_idx], param_name)
    gramian_flat = zeros(
        param.numel(), param.numel(), device=param.device, dtype=param.dtype
    )

    for x_n in X.split(1):
        # manually compute the Laplacian
        activations = manual_forward(layers, x_n)
        gradients = manual_backward(layers, activations)
        hessians = manual_hessian_backward(layers, activations, gradients)
        laplacian = einsum(hessians[0], "batch d d ->")

        # compute the Laplacian's gradient with autodiff
        grad_laplacian = autograd.grad(
            laplacian,
            param,
            allow_unused=True,  # for bias terms
        )[0]

        # The last layer's bias is not used in HBP, hence its  contribution to the
        # Gramian is zero. Autograd will return None for this bias.
        if grad_laplacian is None:
            grad_laplacian = zeros_like(param)

        # form the Gramian
        grad_laplacian_flat = grad_laplacian.detach().flatten()
        gramian_flat += einsum(grad_laplacian_flat, grad_laplacian_flat, "i,j->i j")

    return gramian_flat.reshape(*param.shape, *param.shape)


def main():
    """Compute the Gramian for one weight matrix in a toy MLP.

    We compare three approaches:

    1) Computing the Laplacian manually and the Gramian automatically.

    2) TODO Computing both the Laplacian and Gramian manually.

    3) Computing both the Laplacian and Gramian automatically.
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

    for layer_idx in [0, 2, 4, 6]:
        print(f"Layer {layer_idx} ({layers[layer_idx]})")
        for name in ["weight", "bias"]:
            print(f"\tParameter {name!r}")

            # 1) manual Laplacian, gradients for Gramian via autograd
            gramian1 = manual_laplace_autograd_gramian(layers, X, layer_idx, name)

            # 2) TODO manual Laplacian and Gramian

            # 3) Laplacian and Gramian via autodiff (functorch)
            param_name = f"{layer_idx}.{name}"
            gramian3 = autograd_gramian(model, X, param_name)

            same_1_3 = allclose(gramian1, gramian3)
            print(f"\t\tsame(manual+auto, auto)? {same_1_3}")
            assert same_1_3


if __name__ == "__main__":
    main()
