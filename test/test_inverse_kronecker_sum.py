"""Test class for inverting the sum of two Kronecker products."""

from torch import allclose, eye, manual_seed, rand

from kfac_pinns_exp.inverse_kronecker_sum import InverseKroneckerSum


def test_InverseKroneckerSum_eigh():
    """Test general eigenvalue decomposition matches between backends."""
    manual_seed(0)
    D = 100

    # create symmetric positive definite matrices
    damping = 1e-8 * eye(D).double()
    A = rand(D, D).double()
    A = A @ A.T + damping
    B = rand(D, D).double()
    B = B @ B.T + damping

    # compare their general eigendecompositions among backends
    evals_scipy, evecs_scipy = InverseKroneckerSum.eigh(A, B, backend="scipy")
    evals_torch, evecs_torch = InverseKroneckerSum.eigh(A, B, backend="torch")

    assert allclose(evals_scipy, evals_torch)
    for d in range(D):
        # eigenvectors are ambiguous up to a sign flip
        evec_scipy, evec_torch = evecs_scipy[:, d], evecs_torch[:, d]
        assert allclose(evec_scipy, evec_torch) or allclose(-evec_scipy, evec_torch)
