"""Verify the inverse-of-Kronecker-sum equation using the class interface."""

from torch import allclose, eye, inverse, kron, manual_seed, randn

from kfac_pinns_exp.inverse_kronecker_sum import InverseKroneckerSum


def main():
    """Verify the inverse-of-Kronecker-sum class on a toy problem."""
    manual_seed(0)
    dim1, dim2 = 8, 11
    damping = 1e-4
    tols = {"atol": 1e-7, "rtol": 1e-5}

    # create symmetric positive-definite matrices in double precision
    A = randn(dim1, dim1).double()
    A = A @ A.T + damping * eye(dim1)

    B = randn(dim2, dim2).double()
    B = B @ B.T + damping * eye(dim2)

    C = randn(dim1, dim1).double()
    C = C @ C.T + damping * eye(dim1)

    D = randn(dim2, dim2).double()
    D = D @ D.T + damping * eye(dim2)

    # explicitly compute and invert the matrix
    K = kron(A, B) + kron(C, D)
    K_inv = inverse(K)

    # class approach
    K_inv_class = InverseKroneckerSum(A, B, C, D)

    # multiply onto flattened vector
    x = randn(dim1 * dim2).double()
    assert allclose(K_inv @ x, K_inv_class @ x, **tols)

    # multiply onto un-flattened vector
    x = x.reshape(dim1, dim2)
    assert allclose(K_inv @ x.flatten(), (K_inv_class @ x).flatten(), **tols)


if __name__ == "__main__":
    main()
