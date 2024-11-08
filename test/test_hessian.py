"""Attempt to efficiently compute symmetric Hessian factorizations (for SPRING)."""

from itertools import product

from torch import cat, einsum, eye, manual_seed, rand, zeros, zeros_like
from torch.autograd import grad


def test_hessian():
    """Verify Hessian computed with `grad` versus `is_grads_batched`."""
    N = 10
    D_out = 1
    assert D_out == 1

    manual_seed(0)
    residual = rand(N, D_out, requires_grad=True)
    assert residual.ndim == 2
    loss = 0.5 * residual.pow(2).mean()

    # ground truth
    true_hessian = eye(D_out).expand(N, -1, -1) / N

    # compute the Hessian with autograd + for loop
    loop_hessian = zeros(N, D_out, D_out)

    (gradient,) = grad(loss, residual, create_graph=True)
    for n, d in product(range(N), range(D_out)):
        e_nd = zeros_like(residual)
        e_nd[n, d] = 1

        # compute one column of the n-th samples Hessian
        (hessian_column,) = grad((gradient * e_nd).sum(), residual, retain_graph=True)
        loop_hessian[n, d, :] = hessian_column[n]

    assert loop_hessian.allclose(true_hessian)

    # compute the Hessian with autograd + `is_grads_batched`
    (gradient,) = grad(loss, residual, create_graph=True)

    grad_outputs = eye(D_out).unsqueeze(1).repeat(1, N, 1)

    contracted = einsum("n d, j n d -> n j", gradient, grad_outputs)

    # compute one column of the batched Hessian
    (batched_hessian,) = grad(
        contracted, residual, grad_outputs=grad_outputs, is_grads_batched=True
    )

    assert batched_hessian.allclose(true_hessian)
