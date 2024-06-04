"""Test class for inverting the sum of two Kronecker products."""

from torch import allclose, eye, inverse, kron, manual_seed, rand

from kfac_pinns_exp.inverse_kronecker_sum import InverseKroneckerSum


def test_InverseKroneckerSum__matmul__():
    """Test matrix-vector multiplication with inverse of a Kronecker sum."""
    manual_seed(0)
    D1, D2 = 9, 11

    # create symmetric positive definite matrices
    damping_D1_mat = 1e-8 * eye(D1).double()
    damping_D2_mat = 1e-8 * eye(D2).double()

    A = rand(D1, D1).double()
    A = A @ A.T + damping_D1_mat

    B = rand(D2, D2).double()
    B = B @ B.T + damping_D2_mat

    C = rand(D1, D1).double()
    C = C @ C.T + damping_D1_mat

    D = rand(D2, D2).double()
    D = D @ D.T + damping_D2_mat

    # compute, damp, and invert the Kronecker sum manually
    manual_inv = inverse(kron(A, B) + kron(C, D))
    kronecker_inv = InverseKroneckerSum(A, B, C, D)

    # create a random vector for multiplication and compare matvecs
    x = rand(D1 * D2).double()
    manual_matvec = manual_inv @ x
    kronecker_matvec = kronecker_inv @ x
    assert allclose(manual_matvec, kronecker_matvec)
