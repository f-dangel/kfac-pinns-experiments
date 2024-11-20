"""Test general purpose SPRING for l2 regression."""

from torch import manual_seed, ones, rand
from torch.nn import Linear, Sequential, Sigmoid

from kfac_pinns_exp.optim.spring_standalone import SPRING


def test_spring_standalone():
    """Test if the general purpose spring implementation reduces loss."""
    manual_seed(0)

    # neural network setup
    D_in, D_hidden, D_out = 2, 20, 1
    assert D_out == 1.0, "Atm the autodiff internals require D_out=1"

    net = Sequential(Linear(D_in, D_hidden), Sigmoid(), Linear(D_hidden, D_out))
    params = list(net.parameters())

    # data generation
    N = 10
    X = rand(N, D_in)
    Y = ones(N, D_out)

    # loss and optimizer
    def loss_function(X):
        return 0.5 * (X**2).sum()

    opt = SPRING(params=params, lr=0.01, decay_factor=0.99)

    # training loop
    prev_loss = float("inf")
    for _ in range(0, 10):
        opt.zero_grad()

        def forward():
            residual = net(X) - Y
            assert residual.shape == (N, 1)

            loss = loss_function(residual)
            return loss, residual

        loss = opt.step(forward=forward).item()

        assert prev_loss >= loss, "Loss is not reduced"

        prev_loss = loss
