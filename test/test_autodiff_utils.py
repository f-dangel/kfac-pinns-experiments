"""Test `kfac_pinns_exp.autodiff_utils`."""

from itertools import product
from test.utils import report_nonclose

from torch import manual_seed, rand, zeros, zeros_like
from torch.autograd import grad
from torch.nn import Linear, Sequential, Sigmoid, Tanh

from kfac_pinns_exp.autodiff_utils import autograd_input_divergence


def test_autograd_input_divergence():
    """Test computation of the divergence with `functorch`."""
    manual_seed(0)

    # setup
    N, S, D = 5, 4, 3
    X = rand(N, S, D, requires_grad=True)
    X.requires_grad_(True)
    model = Sequential(Linear(D, D), Tanh(), Linear(D, D), Sigmoid())

    # compute the divergence with `autograd`
    div_true = zeros(N, 1)

    f_X = model(X)

    for n, d, s in product(range(N), range(D), range(S)):
        e = zeros_like(f_X)
        e[n, s, d] = 1.0
        div_true[n] += grad(f_X, X, grad_outputs=e, retain_graph=True)[0][n, s, d]

    # compute the divergence with `functorch`
    div = autograd_input_divergence(model, X)

    assert div.shape == div_true.shape == (N, 1)
    report_nonclose(div, div_true)


def test_autograd_input_divergence_with_coordinates():
    """Test computation of the divergence with specified coordinates."""
    manual_seed(0)

    # setup
    N, S, D = 5, 4, 3
    X = rand(N, S, D, requires_grad=True)

    # S * D input coordinates, but only S output coordinates
    model = Sequential(Linear(D, D), Tanh(), Linear(D, 1), Sigmoid())

    # compute the divergence with `autograd`
    div_true = zeros(N, 1)

    f_X = model(X)
    coordinates = [0, 5, 3, 8]  # must be < S * D and contain S entries
    assert len(coordinates) == f_X.shape[1:].numel()

    for n in range(N):
        for out, coord in zip(range(len(coordinates)), coordinates):
            e = zeros_like(f_X).flatten(start_dim=1)
            e[n, out] = 1.0
            e = e.reshape_as(f_X)
            div_true[n] += grad(f_X, X, grad_outputs=e, retain_graph=True)[0].flatten(
                start_dim=1
            )[n, coord]

    # compute the divergence with `functorch`
    div = autograd_input_divergence(model, X, coordinates=coordinates)

    assert div.shape == div_true.shape == (N, 1)
    report_nonclose(div, div_true)
