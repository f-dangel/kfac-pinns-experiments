"""Test helper functions to solve the Poisson equation."""

from torch import allclose, kron, zeros_like
from torch.nn import Linear, Sequential, Tanh

from kfac_pinns_exp.kfac_utils import gramian_basis_to_kfac_basis
from kfac_pinns_exp.poisson_equation import (
    evaluate_boundary_gramian,
    evaluate_boundary_loss_and_kfac_expand,
    square_boundary,
    u,
)


def test_boundary_kfac_batch_size_1():
    """Compare KFAC and Gramian of the boundary loss for batch size 1.

    In this case, KFAC is exact.
    """
    # everything in double precision
    # data
    N_dOmega, dim_Omega = 1, 2
    X_dOmega = square_boundary(N_dOmega, dim_Omega).double()
    y_dOmega = zeros_like(u(X_dOmega))

    # neural net
    D_hidden = 64
    layers = [Linear(dim_Omega, D_hidden), Tanh(), Linear(D_hidden, 1)]
    layers = [layer.double() for layer in layers]
    model = Sequential(*layers)

    # compute boundary KFACs and Gramians
    _, kfacs = evaluate_boundary_loss_and_kfac_expand(layers, X_dOmega, y_dOmega)
    gramians = evaluate_boundary_gramian(model, X_dOmega, "per_layer")

    # The Gramian's basis is `(W.flatten().T, b.T).T`, but KFAC's basis is
    # `(W, b).flatten()` which is different. Hence, we need to rearrange the Gramian to
    # the basis of KFAC.
    for idx, (A, B) in enumerate(kfacs.values()):
        D_in, D_out = A.shape[0] - 1, B.shape[0]
        gramians[idx] = gramian_basis_to_kfac_basis(gramians[idx], D_in, D_out)

    # Compare Gramian and KFAC
    for gramian, (A, B) in zip(gramians, kfacs.values()):
        assert allclose(gramian, kron(B, A))
