"""Test `kfac_pinns_exp.autodiff_utils`."""

from pytest import mark
from torch import allclose, cat, manual_seed, outer, rand, zeros, zeros_like
from torch.autograd import grad
from torch.nn import Linear, Sequential, Tanh

from kfac_pinns_exp.autodiff_utils import autograd_gramian

LOSS_TYPES = ["boundary", "interior"]
APPROXIMATIONS = ["full", "diagonal", "per_layer"]


@mark.parametrize("approximation", APPROXIMATIONS, ids=APPROXIMATIONS)
@mark.parametrize("loss_type", LOSS_TYPES, ids=LOSS_TYPES)
def test_autograd_gramian(loss_type: str, approximation: str):
    """Test `autograd_gramian`.

    Args:
        loss_type: The type of loss function whose Gramian
            is tested. Can be either `'boundary'` or `'interior`.
        approximation: The type of approximation to the Gramian.
            Can be either `'full'`, `'diagonal'`, or `'per_layer'`.

    Raises:
        ValueError: If `loss_type` is not one of `'boundary'` or `'interior'`.
        ValueError: If `approximation` is not one of `'full'` or `'diagonal'`.
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
    gramian = autograd_gramian(
        model, X, param_names, loss_type=loss_type, approximation=approximation
    )

    # compute the Gramian naively via a for-loop and autograd
    dim = sum(p.numel() for p in params)
    truth = zeros(dim, dim)

    # compute the Gram gradient for sample n and add its contribution
    # to the Gramian
    for n in range(batch_size):
        X_n = X[n].requires_grad_(loss_type == "interior")
        output = model(X_n)

        if loss_type == "boundary":
            gram_grad = grad(output, params)

        elif loss_type == "interior":
            laplace = zeros(())

            for d in range(D_in):
                (grad_input,) = grad(output, X_n, create_graph=True)
                e_d = zeros_like(X_n)
                e_d[d] = 1.0

                (hess_input_dd,) = grad(
                    (e_d * grad_input).sum(), X_n, create_graph=True
                )
                laplace += hess_input_dd[d]

            gram_grad = grad(
                laplace,
                params,
                retain_graph=True,
                # set gradients of un-used parameters to zero
                # (e.g. last layer bias does not affect Laplacian)
                materialize_grads=True,
            )
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # flatten and take the outer product
        gram_grad = cat([g.flatten() for g in gram_grad])
        truth.add_(outer(gram_grad, gram_grad))

    # account for approximation
    if approximation == "full":
        pass
    elif approximation == "diagonal":
        truth = truth.diag()
    elif approximation == "per-layer":
        # compute per-layer number of parameters
        sizes = []
        for layer in model.modules():
            if not list(layer.children()) and list(layer.parameters()):
                sizes.append(sum(p.numel() for p in layer.parameters()))
        # cut the Gramian into per-layer blocks
        truth = [
            row_block.split(sizes, dim=1) for row_block in truth.split(sizes, dim=0)
        ]
        truth = [truth[b][b] for b in range(len(sizes))]

    else:
        raise ValueError(f"Unknown approximation: {approximation}")

    if approximation == "per-layer":
        for b in range(len(truth)):
            assert allclose(gramian[b], truth[b])
    elif approximation in ["diagonal", "full"]:
        assert allclose(gramian, truth)
