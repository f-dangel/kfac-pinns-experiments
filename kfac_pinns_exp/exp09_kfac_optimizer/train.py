"""Pseudo-code for the training loop with KFAC for PINNs"""

from typing import List

from torch.nn import Module

from kfac_pinns_exp.exp09_kfac_optimizer.optimizer import KFACForPINNs

layers: List[Module] = [...]

T_kfac = 1
T_inv = 1
ema_factor = 0.95  # exponentially moving average over KFAC factors
damping = 1e-3
lr = 1e-2

optimizer = KFACForPINNs(
    layers, lr, damping, T_kfac=T_kfac, T_inv=T_inv, ema_factor=ema_factor
)

batch_size_interior = 256
batch_size_boundary = 128

num_steps = 1_000

for _ in range(num_steps):
    optimizer.zero_grad()

    # [N, dim(Ω)] (= x), [N, 1] (= - f(x))
    X_interior, y_interior = get_interior_batch(batch_size_interior)
    # depending on step, either updates the KFAC approximation and the loss, or
    # only computes the loss
    loss_interior = optimizer.evaluate_interior_loss_and_update_kfac(
        X_interior, y_interior
    )

    # compute the interior loss' gradient
    loss_interior.backward()

    # [N, dim(Ω)] (= x), [N, 1] (= +g(x))
    X_boundary, y_boundary = get_boundary_batch(batch_size_boundary)
    # depending on step, either updates the KFAC approximation and the loss, or
    # only computes the loss
    loss_boundary = optimizer.evaluate_boundary_loss_and_update_kfac(
        X_boundary, y_boundary
    )

    # compute the boundary loss' gradient
    loss_boundary.backward()

    # depending on step, maybe update the inverse curvature, compute the
    # pre-conditioned gradient and update the parameters, increment step internally
    optimizer.step()
