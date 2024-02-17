"""Test `kfac_pinns_exp.autodiff_utils`."""

from torch import allclose, cat, manual_seed, outer, rand, zeros
from torch.autograd import grad
from torch.nn import Linear, Sequential, Tanh

from kfac_pinns_exp.autodiff_utils import autograd_gramian


def test_autograd_gramian():
    """Test `autograd_gramian`.

    Only tests the boundary Gramian. The interior Gramian is tested in
    `exp01`.
    """
    manual_seed(0)
    # hyper-parametersj
    D_in, D_hidden, D_out = 3, 10, 1
    batch_size = 5
    assert D_out == 1

    # set up data and model
    X = rand(batch_size, D_in)
    model = Sequential(
        Linear(D_in, D_hidden),
        Tanh(),
        Linear(D_hidden, D_hidden),
        Tanh(),
        Linear(D_hidden, D_out),
    )

    # compute the boundary Gramian with functorch
    params = list(model.parameters())
    param_names = [n for n, _ in model.named_parameters()]
    loss_type = "boundary"
    gramian = autograd_gramian(model, X, param_names, loss_type=loss_type)

    # compute the boundary Gramian with autograd
    dim = sum(p.numel() for p in params)
    truth = zeros(dim, dim)

    for n in range(batch_size):
        output = model(X[n])
        grad_output = grad(output, params)
        grad_output = cat([g.flatten() for g in grad_output])
        truth.add_(outer(grad_output, grad_output))

    assert allclose(gramian, truth)
