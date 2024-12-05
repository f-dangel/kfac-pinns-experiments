"""Test general purpose SPRING for l2 regression."""

from test.utils import DEVICE_IDS, DEVICES

from pytest import mark
from torch import device, dtype, float64, manual_seed, ones, rand
from torch.nn import Linear, Sequential, Sigmoid

from kfac_pinns_exp.optim.spring_standalone import SPRING


@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
def test_spring_standalone(device: device, dtype: dtype = float64):
    """Test if the general purpose spring implementation reduces loss.

    Args:
        device: The devices the optimizer will run on, cpu and gpu.
        dtype: The type of tensors used. Default: `float64`.
    """
    manual_seed(0)

    # neural network setup
    D_in, D_hidden, D_out = 2, 20, 1
    assert D_out == 1, "Atm the autodiff internals require D_out=1"

    layers = [Linear(D_in, D_hidden), Sigmoid(), Linear(D_hidden, D_out)]
    net = Sequential(*layers).to(device, dtype)
    params = list(net.parameters())

    # data generation
    N = 10
    X = rand(N, D_in).to(device, dtype)
    Y = ones(N, D_out).to(device, dtype)

    # loss and optimizer
    def loss_function(X):
        return 0.5 * (X**2).sum()

    opt = SPRING(params=params, lr=0.01)

    # training loop
    prev_loss = float("inf")
    for _ in range(10):
        opt.zero_grad()

        def forward():
            residual = net(X) - Y
            assert residual.shape == (N, 1)

            loss = loss_function(residual)
            return loss, residual

        loss = opt.step(forward=forward).item()

        assert prev_loss >= loss, "Loss is not reduced"

        prev_loss = loss
