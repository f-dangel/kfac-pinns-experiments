"""Demonstrate Laplacian computation on a small toy MLP."""


from einops import einsum
from torch import allclose, manual_seed, rand
from torch.nn import Linear, Sequential, Sigmoid

from kfac_pinns_exp.autodiff_utils import autograd_input_hessian
from kfac_pinns_exp.manual_differentiation import (
    manual_backward,
    manual_forward,
    manual_hessian_backward,
)


def main():
    """Compute and compare the input Hessian and Laplacian on a toy MLP."""
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

    # manual computation (via Hessian backpropagation)
    activations = manual_forward(layers, X)
    gradients = manual_backward(layers, activations)
    hessians = manual_hessian_backward(layers, activations, gradients)
    hessian_X = hessians[0]
    laplacian_X = einsum(hessian_X, "batch d d ->")

    # automatic computation (via functorch)
    true_hessian_X = autograd_input_hessian(Sequential(*layers), X)
    true_laplacian_X = einsum(true_hessian_X, "batch d d ->")

    # compare
    print("Manual = autograd?")
    same_hessian_X = allclose(hessian_X, true_hessian_X)
    print(f"\thessian_X: {same_hessian_X}")

    same_laplacian_X = allclose(laplacian_X, true_laplacian_X)
    print(f"\tlaplacian_X: {same_laplacian_X}")

    assert same_hessian_X
    assert same_laplacian_X


if __name__ == "__main__":
    main()
