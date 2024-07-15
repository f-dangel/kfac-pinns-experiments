"""Test helper functions to solve the Poisson equation."""

from test.utils import report_nonclose

from torch import kron, manual_seed, rand, tensor, zeros_like
from torch.nn import Linear, Sequential, Tanh

from kfac_pinns_exp.gramian_utils import autograd_gramian
from kfac_pinns_exp.kfac_utils import gramian_basis_to_kfac_basis
from kfac_pinns_exp.pinn_utils import evaluate_boundary_loss_and_kfac
from kfac_pinns_exp.poisson_equation import (
    evaluate_interior_loss_and_kfac,
    f_sin_product,
    square_boundary,
    u_sin_product,
)


def test_boundary_kfac_batch_size_1():
    """Compare KFAC and Gramian of the boundary loss for batch size 1.

    In this case, KFAC is exact.
    """
    # everything in double precision
    # data
    N_dOmega, dim_Omega = 1, 2
    X_dOmega = square_boundary(N_dOmega, dim_Omega).double()
    y_dOmega = zeros_like(u_sin_product(X_dOmega))

    # neural net
    D_hidden = 64
    layers = [Linear(dim_Omega, D_hidden), Tanh(), Linear(D_hidden, 1)]
    layers = [layer.double() for layer in layers]
    model = Sequential(*layers)

    # compute boundary KFACs and Gramians
    _, kfacs = evaluate_boundary_loss_and_kfac(layers, X_dOmega, y_dOmega)
    # NOTE For batch size 1 we don't have to divide by the batch size explicitly
    gramians = autograd_gramian(
        model,
        X_dOmega,
        [n for n, _ in model.named_parameters()],
        loss_type="poisson_boundary",
        approximation="per_layer",
    )

    # The Gramian's basis is `(W.flatten().T, b.T).T`, but KFAC's basis is
    # `(W, b).flatten()` which is different. Hence, we need to rearrange the Gramian to
    # the basis of KFAC.
    for idx, (A, B) in enumerate(kfacs.values()):
        D_in, D_out = A.shape[0] - 1, B.shape[0]
        gramians[idx] = gramian_basis_to_kfac_basis(gramians[idx], D_in, D_out)

    # Compare Gramian and KFAC
    for gramian, (A, B) in zip(gramians, kfacs.values()):
        report_nonclose(gramian, kron(B, A))


def test_evaluate_interior_loss_and_kfac():
    """Make sure the interior KFAC computation does not change over time.

    We computed KFAC on a commit we trust, because the related optimizer showed good
    performance. We now want to make sure that the KFAC computation does not change
    over time, because it is crucial for the optimizer to work correctly.

    Commit: dbf432d6acc6d40e308500bf96d662213d0d919a
    Hardware: MacBook Pro M2, PyTorch 2.2.0, Python 3.9.16
    """
    manual_seed(0)
    dim_Omega = 2
    layers = [Linear(dim_Omega, 4), Tanh(), Linear(4, 3), Tanh(), Linear(3, 1)]
    tols = {"rtol": 2e-5}
    N = 20
    X = rand(N, dim_Omega)
    y = f_sin_product(X)

    loss, kfacs = evaluate_interior_loss_and_kfac(layers, X, y)
    assert list(kfacs.keys()) == [0, 2, 4]
    report_nonclose(loss, tensor(43.3608), **tols)

    # first Linear KFAC
    A_0 = tensor(
        [
            [3.40178e-01, 6.50632e-02, 1.27614e-01],
            [6.50632e-02, 3.29262e-01, 1.23122e-01],
            [1.27614e-01, 1.23122e-01, 2.50000e-01],
        ]
    )
    B_0 = tensor(
        [
            [2.89014e-02, 3.20101e-03, 3.92351e-02, 4.51581e-03],
            [3.20101e-03, 1.29344e-03, 5.11732e-03, 6.91636e-04],
            [3.92351e-02, 5.11732e-03, 6.24850e-02, 6.72605e-03],
            [4.51581e-03, 6.91636e-04, 6.72605e-03, 9.97410e-04],
        ]
    )
    report_nonclose(kfacs[0][0], A_0, **tols)
    report_nonclose(kfacs[0][1], B_0, **tols)

    # second Linear KFAC
    A_2 = tensor(
        [
            [4.08669e-02, -5.69003e-02, 9.75280e-03, 5.87176e-02, 2.98905e-02],
            [-5.69003e-02, 1.83456e-01, 3.87553e-02, -8.15583e-02, -8.37203e-02],
            [9.75280e-03, 3.87553e-02, 4.22566e-02, 1.73881e-02, -6.28469e-02],
            [5.87176e-02, -8.15583e-02, 1.73881e-02, 8.56563e-02, 3.16305e-02],
            [2.98905e-02, -8.37203e-02, -6.28469e-02, 3.16305e-02, 2.50000e-01],
        ]
    )
    B_2 = tensor(
        [
            [2.87340e-01, 1.94413e-01, 6.63347e-02],
            [1.94413e-01, 1.33150e-01, 4.54165e-02],
            [6.63347e-02, 4.54165e-02, 1.70896e-02],
        ]
    )
    report_nonclose(kfacs[2][0], A_2, **tols)
    report_nonclose(kfacs[2][1], B_2, **tols)

    # fourth Linear KFAC
    A_4 = tensor(
        [
            [1.76463e-02, -3.09704e-03, -1.00101e-02, 6.79312e-03],
            [-3.09704e-03, 1.93018e-02, 4.27060e-03, 8.37654e-03],
            [-1.00101e-02, 4.27060e-03, 4.67343e-02, 9.65116e-02],
            [6.79312e-03, 8.37654e-03, 9.65116e-02, 2.50000e-01],
        ]
    )
    B_4 = tensor([[1.00000e00]])
    report_nonclose(kfacs[4][0], A_4, **tols)
    report_nonclose(kfacs[4][1], B_4, **tols)


def test_evaluate_boundary_loss_and_kfac():
    """Make sure the boundary KFAC computation does not change over time.

    We computed KFAC on a commit we trust, because the related optimizer showed good
    performance. We now want to make sure that the KFAC computation does not change
    over time, because it is crucial for the optimizer to work correctly.

    Commit: dbf432d6acc6d40e308500bf96d662213d0d919a
    Hardware: MacBook Pro M2, PyTorch 2.2.0, Python 3.9.16
    """
    manual_seed(0)
    dim_Omega = 2
    layers = [Linear(dim_Omega, 4), Tanh(), Linear(4, 3), Tanh(), Linear(3, 1)]
    tols = {"rtol": 2e-5}
    N = 10
    X = square_boundary(N, dim_Omega)
    y = u_sin_product(X)

    loss, kfacs = evaluate_boundary_loss_and_kfac(layers, X, y)
    assert list(kfacs.keys()) == [0, 2, 4]
    report_nonclose(loss, tensor(4.71314e-2), **tols)

    # first Linear KFAC
    A_0 = tensor(
        [
            [2.72772e-01, 1.68159e-01, 3.68213e-01],
            [1.68159e-01, 3.96122e-01, 4.48048e-01],
            [3.68213e-01, 4.48048e-01, 1.00000e00],
        ]
    )
    B_0 = tensor(
        [
            [2.41287e-02, 2.52728e-03, 3.53685e-02, 3.29621e-03],
            [2.52728e-03, 2.86675e-04, 3.71841e-03, 3.53295e-04],
            [3.53685e-02, 3.71841e-03, 5.22259e-02, 4.86296e-03],
            [3.29621e-03, 3.53295e-04, 4.86296e-03, 4.56748e-04],
        ]
    )
    report_nonclose(kfacs[0][0], A_0, **tols)
    report_nonclose(kfacs[0][1], B_0, **tols)

    # second Linear KFAC
    A_2 = tensor(
        [
            [3.70828e-02, -5.59039e-02, -9.82343e-03, 4.90909e-02, 1.01751e-01],
            [-5.59039e-02, 1.29339e-01, 5.29436e-02, -6.97071e-02, -2.30640e-01],
            [-9.82343e-03, 5.29436e-02, 6.33693e-02, -3.29883e-03, -2.21240e-01],
            [4.90909e-02, -6.97071e-02, -3.29883e-03, 6.66986e-02, 9.96794e-02],
            [1.01751e-01, -2.30640e-01, -2.21240e-01, 9.96794e-02, 1.00000e00],
        ]
    )
    B_2 = tensor(
        [
            [2.82707e-01, 1.90379e-01, 6.44314e-02],
            [1.90379e-01, 1.28231e-01, 4.33994e-02],
            [6.44314e-02, 4.33994e-02, 1.47452e-02],
        ]
    )
    report_nonclose(kfacs[2][0], A_2, **tols)
    report_nonclose(kfacs[2][1], B_2, **tols)

    # third Linear KFAC
    A_4 = tensor(
        [
            [7.28320e-03, 7.57422e-05, -7.76281e-03, -5.47431e-03],
            [7.57422e-05, 1.48446e-02, 2.20520e-02, 5.64785e-02],
            [-7.76281e-03, 2.20520e-02, 1.70723e-01, 4.07970e-01],
            [-5.47431e-03, 5.64785e-02, 4.07970e-01, 1.00000e00],
        ]
    )
    B_4 = tensor([[1.00000e00]])
    report_nonclose(kfacs[4][0], A_4, **tols)
    report_nonclose(kfacs[4][1], B_4, **tols)
