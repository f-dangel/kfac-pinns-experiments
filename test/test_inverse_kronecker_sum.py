"""Test class for inverting the sum of two Kronecker products."""

from pytest import mark
from torch import allclose, eye, inverse, kron, manual_seed, rand

from kfac_pinns_exp.inverse_kronecker_sum import InverseKroneckerSum


def test_InverseKroneckerSum_eigh():
    """Test general eigenvalue decomposition matches between backends."""
    manual_seed(0)
    D = 100

    # create symmetric positive definite matrices
    damping_mat = 1e-8 * eye(D).double()
    A = rand(D, D).double()
    A = A @ A.T + damping_mat
    B = rand(D, D).double()
    B = B @ B.T + damping_mat

    # compare their general eigendecompositions among backends
    evals_scipy, evecs_scipy, B_inv_scipy = InverseKroneckerSum.eigh(
        A, B, backend="scipy", return_B_inv=True
    )
    evals_torch, evecs_torch, B_inv_torch = InverseKroneckerSum.eigh(
        A, B, backend="torch", return_B_inv=True
    )

    assert allclose(evals_scipy, evals_torch)
    for d in range(D):
        # eigenvectors are ambiguous up to a sign flip
        evec_scipy, evec_torch = evecs_scipy[:, d], evecs_torch[:, d]
        assert allclose(evec_scipy, evec_torch) or allclose(-evec_scipy, evec_torch)
    assert allclose(B_inv_scipy, B_inv_torch)


BACKENDS = ["scipy", "torch"]
BACKEND_IDS = [f"backend={backend}" for backend in BACKENDS]


@mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
def test_InverseKroneckerSum__matmul__(backend: str):
    """Test matrix-vector multiplication with inverse of a Kronecker sum.

    Args:
        backend: Backend for the eigendecomposition.
    """
    manual_seed(0)
    D1, D2 = 9, 11

    # create symmetric positive definite matrices
    damping_D1_mat = 1e-8 * eye(D1).double()
    damping_D2_mat = 1e-8 * eye(D2).double()

    A1 = rand(D1, D1).double()
    A1 = A1 @ A1.T + damping_D1_mat

    A2 = rand(D2, D2).double()
    A2 = A2 @ A2.T + damping_D2_mat

    B1 = rand(D1, D1).double()
    B1 = B1 @ B1.T + damping_D1_mat

    B2 = rand(D2, D2).double()
    B2 = B2 @ B2.T + damping_D2_mat

    # compute, damp, and invert the Kronecker sum manually
    manual_inv = inverse(kron(A1, A2) + kron(B1, B2))
    kronecker_inv = InverseKroneckerSum(A1, A2, B1, B2, backend=backend)

    # create a random vector for multiplication and compare matvecs
    x = rand(D1 * D2).double()
    manual_matvec = manual_inv @ x
    kronecker_matvec = kronecker_inv @ x
    assert allclose(manual_matvec, kronecker_matvec)
