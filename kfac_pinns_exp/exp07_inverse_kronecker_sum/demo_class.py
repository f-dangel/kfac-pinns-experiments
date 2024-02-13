"""Verify the inverse-of-Kronecker-sum equation using the class interface."""

from torch import allclose, eye, inverse, kron, manual_seed, randn

from kfac_pinns_exp.exp07_inverse_kronecker_sum.inverse_kronecker_sum import (
    InverseKroneckerSum,
)


def main():
    """Verify the inverse-of-Kronecker-sum class on a toy problem."""
    manual_seed(0)
    dim1, dim2 = 8, 11
    damping = 1e-4
    tols = {"atol": 1e-7, "rtol": 1e-5}

    # create symmetric positive-definite matrices in double precision
    A1 = randn(dim1, dim1).double()
    A1 = A1 @ A1.T + damping * eye(dim1)

    A2 = randn(dim2, dim2).double()
    A2 = A2 @ A2.T + damping * eye(dim2)

    B1 = randn(dim1, dim1).double()
    B1 = B1 @ B1.T + damping * eye(dim1)

    B2 = randn(dim2, dim2).double()
    B2 = B2 @ B2.T + damping * eye(dim2)

    # explicitly compute and invert the matrix
    K = kron(A1, A2) + kron(B1, B2)
    K_inv = inverse(K)

    # class approach
    K_inv_class = InverseKroneckerSum(A1, A2, B1, B2)

    # multiply onto flattened vector
    x = randn(dim1 * dim2).double()
    assert allclose(K_inv @ x, K_inv_class @ x, **tols)

    # multiply onto un-flattened vector
    x = x.reshape(dim1, dim2)
    assert allclose(K_inv @ x.flatten(), (K_inv_class @ x).flatten(), **tols)


if __name__ == "__main__":
    main()
