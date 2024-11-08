"""Attempt to efficiently compute Jacobians (needed for SPRING)."""

from time import time

from torch import allclose, cat, eye, manual_seed, ones_like, rand, zeros
from torch.autograd import grad
from torch.nn import Linear, Sequential, Sigmoid

from kfac_pinns_exp.optim.spring_standalone import SPRING


def test_spring_standalone():
    """Verify Jacobian computed with `grad` + `is_grads_batched`."""
    D_in, D_hidden, D_out = 1, 20, 1
    assert D_out == 1
    N1 = 10
    N2 = 2
    N = N1 + N2

    manual_seed(0)

    net = Sequential(
        Linear(D_in, D_hidden),
        Sigmoid(),
        Linear(D_hidden, D_out),
    )

    params = list(net.parameters())
    P = sum(p.numel() for p in params)

    def loss_function(X, Y):
        return 0.5 * ((X - Y) ** 2).mean()

    opt = SPRING(params=params, lr=0.05)

    for step_idx in range(0, 1000):
        X1 = rand(N1, D_in)
        X2 = rand(N2, D_in)

        Y1 = ones_like(X1)
        Y2 = ones_like(X2)
        Y = cat([Y1, Y2])  # (N, d_out)

        opt.zero_grad()

        def forward():
            r1 = net(X1)
            r2 = net(X2)

            residual = cat([r1, r2])  # shape (N, d_out)

            loss = loss_function(residual, Y)
            return loss, residual

        loss, _ = forward()

        print(opt.step(forward=forward).item())


if __name__ == "__main__":
    test_spring_standalone()
