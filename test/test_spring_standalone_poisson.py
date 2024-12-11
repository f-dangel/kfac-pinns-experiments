"""Test Spring for a 2d Poisson equation."""

from math import sqrt
from test.utils import DEVICE_IDS, DEVICES

from pytest import mark
from torch import cat, device, dtype, float64, manual_seed, rand
from torch.nn import Linear, Sequential, Tanh

from kfac_pinns_exp.optim.spring_standalone import SPRING
from kfac_pinns_exp.poisson_equation import (
    evaluate_interior_loss,
    f_sin_product,
    square_boundary,
)


@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
def test_spring_standalone(device: device, dtype: dtype = float64):
    """Test if the general purpose spring implementation reduces loss.

    Args:
        device: The devices the optimizer will run on, cpu and gpu.
        dtype: The type of tensors used. Default: `float64`.
    """
    manual_seed(0)

    # Batchsize, physical dimension etc
    N_Omega = 500
    N_Gamma = 200
    d = 2

    # neural network setup
    layers = [Linear(d, 48), Tanh(), Linear(48, 1)]
    net = Sequential(*layers).to(device, dtype)
    params = list(net.parameters())

    # collocation points
    X_Omega = rand(N_Omega, d).to(device, dtype)
    X_Gamma = square_boundary(N_Gamma, d).to(device, dtype)

    Y_Omega = f_sin_product(X_Omega).to(device, dtype)
    Y_Gamma = f_sin_product(X_Gamma).to(device, dtype)

    def interior_residual(model, X, Y):
        _, residual, _ = evaluate_interior_loss(model, X, Y)
        return 1.0 / sqrt(N_Omega) * residual

    def boundary_residual(model, X, Y):
        return 1.0 / sqrt(N_Gamma) * (model(X) - Y)

    assert interior_residual(net, X_Omega, Y_Omega).shape == (N_Omega, 1)
    assert boundary_residual(net, X_Gamma, Y_Gamma).shape == (N_Gamma, 1)

    opt = SPRING(params=params, lr=0.05)

    # training loop
    prev_loss = float("inf")
    for step_idx in range(0, 2):
        opt.zero_grad()

        def forward():
            r_1 = interior_residual(net, X_Omega, Y_Omega)
            r_2 = boundary_residual(net, X_Gamma, Y_Gamma)
            r = cat([r_1, r_2])

            loss = 0.5 * (r**2).sum()
            return loss, r

        loss = opt.step(forward=forward).item()

        print(step_idx, loss)

        assert prev_loss >= loss, "Loss is not reduced"

        prev_loss = loss
