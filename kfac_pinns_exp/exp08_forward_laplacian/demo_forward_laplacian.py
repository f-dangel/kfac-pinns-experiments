"""Demonstrate how to use the forward Laplacian framework."""

from einops import einsum
from torch import allclose, manual_seed, rand
from torch.nn import Linear, Sequential, Sigmoid

from kfac_pinns_exp.autodiff_utils import autograd_input_hessian
from kfac_pinns_exp.forward_laplacian import manual_forward_laplacian


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
