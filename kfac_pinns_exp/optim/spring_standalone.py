"""First draft for a general purpose SPRING implementation."""

from math import sqrt
from typing import Callable, List, Tuple

from torch import Tensor, arange, cat, cholesky_solve, eye, no_grad, zeros_like
from torch.autograd import grad
from torch.linalg import cholesky
from torch.optim import Optimizer


class SPRING(Optimizer):
    """SPRING general purpose optimizer.

    See https://arxiv.org/pdf/2401.10190 for details.
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float,
        damping: float = 1e-3,
        adaptive_damping: bool = False,
        decay_factor: float = 0.99,
        norm_constraint: float = 1e-3,
    ):
        """Set up the SPRING optimizer.

        Args:
            params: The trainable params of the neural network.
            lr: The learning rate.
            damping: The non-negative damping factor (λ in the paper).
                Default: `1e-3` (taken from Section 4 of the paper).
            adaptive_damping: If adapt_damping is True, then the damping parameter is
                adapted according to a trust-region strategy. See Section 4.1 in
                https://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf
                Default: False (no adaptive damping).
            decay_factor: The decay factor (μ in the paper). Must be in `[0; 1)`.
                Default: `0.99` (taken from Section 4 of the paper).
            norm_constraint: The positive norm constraint (C in the paper).
                Default: `1e-3` (taken from Section 4 of the paper).

        Raises:
            ValueError: If the optimizer is used with per-parameter options.
        """
        defaults = dict(
            lr=lr,
            damping=damping,
            decay_factor=decay_factor,
            norm_constraint=norm_constraint,
        )
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("SPRING does not support per-parameter options.")

        self.steps = 0
        self.adaptive_damping = adaptive_damping

        # initialize phi
        (group,) = self.param_groups
        for p in group["params"]:
            self.state[p]["phi"] = zeros_like(p)

    def step(
        self,
        forward: Callable[[], Tuple[Tensor, Tensor]],
    ) -> Tensor:
        """Perform a parameter update step.

        Args:
            forward: A function that computes both the residual and the loss and
                returns both. Note that unlike the PyTorch API of LBFGS, the
                forward does not compute gradients, hence it should not zero the
                gradients either. Instead it splits the model and the
                loss at the `linearization point'. Here is a pseudo-code
                example for how it should look in the training-loop:
                ```
                for _ in range(iterations):

                    X, Y = get_data()

                    def forward():
                        residual = model(X)
                        loss = loss_function(residual, Y)
                        return loss, residual

                    optimizer.step(forward=forward)
                ```

        Returns:
            The loss before the optimization step is taken.

        Raises:
            ValueError: If the residual returned by `forward` is not of shape `(N, 1)`.
        """
        (group,) = self.param_groups
        params = group["params"]
        sizes = [p.numel() for p in params]
        lr = group["lr"]
        damping = group["damping"]
        decay_factor = group["decay_factor"]
        norm_constraint = group["norm_constraint"]

        # compute J, residual and loss
        loss, residual = forward()
        N = residual.shape[0]
        if residual.shape != (N, 1):
            raise ValueError(
                f"The current implementation assumes that the residual is "
                f"of shape {(N, 1)} but is {residual.shape}."
            )

        grad_outputs = eye(N).unsqueeze(-1)
        J = grad(residual, params, grad_outputs=grad_outputs, is_grads_batched=True)
        J = cat([j.flatten(start_dim=1) for j in J], dim=1)

        # compute zeta
        J_phi = J @ cat([self.state[p]["phi"].flatten() for p in params]).unsqueeze(-1)
        zeta = residual + J_phi.mul_(decay_factor)

        # compute preconditioner
        JJT = (J @ J.T).detach()
        idx = arange(JJT.shape[0], device=JJT.device)
        JJT[idx, idx] = JJT.diag() + damping

        # compute step
        step = cholesky_solve(zeta, cholesky(JJT))
        step = -(J.T @ step).squeeze()
        step = [s.reshape_as(p) for s, p in zip(step.split(sizes), params)]

        # update phi
        for p, s in zip(params, step):
            self.state[p]["phi"].mul_(decay_factor).add_(s)

        # compute effective learning rate
        norm_phi = sum([(self.state[p]["phi"] ** 2).sum() for p in params]).sqrt()
        scale = min(lr, (sqrt(norm_constraint) / norm_phi).item())

        # update parameters
        for p in params:
            p.data.add_(self.state[p]["phi"], alpha=scale)

        if self.adaptive_damping:
            # compute quadratic model at previous and new iterate
            q = 0.5 * (zeta**2).sum()
            step = cat([s.flatten() for s in step])
            q_new = 0.5 * ((J @ step + zeta.squeeze()) ** 2).sum()

            # compute loss at new parameter value
            with no_grad():
                loss_new, _ = forward()

            # compute reduction ratio
            rho = (loss_new - loss) / (q_new - q)

            # adapt trust-region
            if rho < 0.25:
                group["damping"] *= 3 / 2
            elif rho > 0.75:
                group["damping"] *= 2 / 3

        self.steps += 1
        return loss
