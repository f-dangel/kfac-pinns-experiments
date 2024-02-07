"""Verify the inverse-of-Kronecker-sum equation."""

from einops import einsum
from scipy.linalg import eigh
from torch import allclose, eye, from_numpy, inverse, kron, manual_seed, randn


def main():
    """Verify the inverse-of-Kronecker-sum equation on a toy problem."""
    manual_seed(0)
    dim = 10
    damping = 1e-4

    # create symmetric positive-definite matrices in double precision
    A1 = randn(dim, dim).double()
    A1 = A1 @ A1.T + damping * eye(dim)

    A2 = randn(dim, dim).double()
    A2 = A2 @ A2.T + damping * eye(dim)

    B1 = randn(dim, dim).double()
    B1 = B1 @ B1.T + damping * eye(dim)

    B2 = randn(dim, dim).double()
    B2 = B2 @ B2.T + damping * eye(dim)

    # explicitly compute and invert the matrix
    K = kron(A1, A2) + kron(B1, B2)
    K_inv = inverse(K)

    # manual approach
    diag1, V1 = eigh(A1.numpy(), B1.numpy())
    diag2, V2 = eigh(A2.numpy(), B2.numpy())

    diag1, V1 = from_numpy(diag1), from_numpy(V1)
    diag2, V2 = from_numpy(diag2), from_numpy(V2)

    V1_inv, V2_inv = inverse(V1), inverse(V2)
    B1_inv, B2_inv = inverse(B1), inverse(B2)

    diag12 = einsum(diag1, diag2, "i,j->i j").flatten() + 1

    K_inv_manual = (
        kron(V1, V2) @ (1 / diag12).diag() @ kron(V1_inv @ B1_inv, V2_inv @ B2_inv)
    )

    assert allclose(K_inv, K_inv_manual, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    main()
