"""Attempt to efficiently compute symmetric Hessian factorizations (for SPRING)."""

from itertools import product
from math import sqrt

from torch import (
    Tensor,
    cat,
    einsum,
    eye,
    manual_seed,
    rand,
    randint,
    split,
    stack,
    zeros,
    zeros_like,
)
from torch.autograd import grad
from torch.linalg import cholesky
from torch.nn.functional import cross_entropy


def compute_batched_hessian_loop(f: Tensor, x: Tensor) -> Tensor:
    """Compute the batched Hessian of `f` w.r.t. `x` using autograd and a for loop.

    Args:
        f: A scalar tensor whose batched derivative will be computed.
        x: A tensor of shape `[N, *]` where `*` is arbitrary and `N` is the batch size.

    Returns:
        The batched Hessian of `f` w.r.t. `x`, i.e. the stack of
        { ∂²f / ∂x[n]² | n = 1,..., N }. Has shape `[N, *, *]`.

    Raises:
        NotImplementedError: If `x` is not 2d.
        ValueError: If `f` is not a scalar.
    """
    if f.shape != ():
        raise ValueError(f"`f` must be a scalar. Got shape {f.shape}.")
    if x.ndim != 2:
        raise NotImplementedError(f"`x` must be 2d for now. Got shape {x.shape}.")

    N, D = x.shape
    hessian = zeros(N, D, D, device=x.device, dtype=x.dtype)

    (df_dx,) = grad(f, x, create_graph=True)

    for n, d in product(range(N), range(D)):
        # create a one-hot vector
        e_nd = zeros_like(x)
        e_nd[n, d] = 1

        # compute one column of the Hessian
        (d2f_dx_column_nd,) = grad((df_dx * e_nd).sum(), x, retain_graph=True)
        hessian[n, d, :] = d2f_dx_column_nd[n]

    return hessian


def compute_batched_hessian(f: Tensor, x: Tensor) -> Tensor:
    """Compute the batched Hessian of `f` w.r.t. `x` using autograd and vmap.

    Uses the `is_grads_batched` argument.

    Args:
        f: A scalar tensor whose batched derivative will be computed.
        x: A tensor of shape `[N, *]` where `*` is arbitrary and `N` is the batch size.

    Returns:
        The batched Hessian of `f` w.r.t. `x`, i.e. the stack of
        { ∂²f / ∂x[n]² | n = 1,..., N }. Has shape `[N, *, *]`.

    Raises:
        NotImplementedError: If `x` is not 2d.
        ValueError: If `f` is not a scalar.
    """
    if f.shape != ():
        raise ValueError(f"`f` must be a scalar. Got shape {f.shape}.")
    if x.ndim != 2:
        raise NotImplementedError(f"`x` must be 2d for now. Got shape {x.shape}.")

    N, D = x.shape

    (df_dx,) = grad(f, x, create_graph=True)
    grad_outputs = eye(D, device=x.device, dtype=x.dtype).unsqueeze(1).repeat(1, N, 1)
    contracted = einsum("n d, j n d -> n j", df_dx, grad_outputs)

    # compute all columns of the batched Hessian in parallel
    (hessian,) = grad(contracted, x, grad_outputs=grad_outputs, is_grads_batched=True)
    # move batch axis first
    return einsum("i n j -> n i j", hessian)


def test_hessian_and_hessian_sqrt():
    """Verify Hessian computed with `grad` versus `is_grads_batched`.

    Use the PINN setup where the residual is a concatenation, and the
    loss function splits it back into the original parts, then computes
    the MSE independently, giving two different reduction factors.
    """
    N1 = 10
    N2 = 5
    D = 3

    manual_seed(0)
    residual1 = rand(N1, D, requires_grad=True)
    residual2 = rand(N2, D, requires_grad=True)
    residual = cat([residual1, residual2])
    # MSE over the two residuals independently, then sum
    loss = sum(0.5 * r.pow(2).mean() for r in split(residual, [N1, N2]))

    # ground truth
    true_hessian = cat(
        [
            eye(D).expand(N1, -1, -1) / residual1.numel(),
            eye(D).expand(N2, -1, -1) / residual2.numel(),
        ]
    )
    true_hessian_sqrt = cat(
        [
            eye(D).expand(N1, -1, -1) / sqrt(residual1.numel()),
            eye(D).expand(N2, -1, -1) / sqrt(residual2.numel()),
        ]
    )

    # compute the Hessian with autograd + for loop
    loop_hessian = compute_batched_hessian_loop(loss, residual)
    # compute the Hessian with autograd + `is_grads_batched`
    batched_hessian = compute_batched_hessian(loss, residual)

    assert loop_hessian.allclose(true_hessian)
    assert batched_hessian.allclose(true_hessian)

    # compute the symmetric decomposition and verify it squares to the Hessian
    hessian_sqrt = cholesky(batched_hessian)
    assert hessian_sqrt.allclose(true_hessian_sqrt)
    assert true_hessian.allclose(
        einsum("n i j, n k j -> n i k", hessian_sqrt, hessian_sqrt)
    )


def test_hessian_crossentropy():
    """Verify Hessian computed with `grad` versus `is_grads_batched`.

    Use softmax cross-entropy loss, whose Hessian is easy to compute manually.
    """
    N, C = 10, 3

    manual_seed(0)
    residual = rand(N, C, requires_grad=True)
    labels = randint(C, (N,))
    loss = cross_entropy(residual, labels)

    # manual Hessian
    p = residual.softmax(dim=1)
    true_hessian = (stack([row.diag() for row in p]) - einsum("ni,nj->nij", p, p)) / N

    # compute the Hessian with autograd + for loop
    loop_hessian = compute_batched_hessian_loop(loss, residual)

    # compute the Hessian with autograd + `is_grads_batched`
    batched_hessian = compute_batched_hessian(loss, residual)

    assert loop_hessian.allclose(true_hessian)
    assert batched_hessian.allclose(true_hessian)
