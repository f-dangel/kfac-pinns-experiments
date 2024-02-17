"""Implement the baseline optimizer that uses the exact Gramian."""

from argparse import ArgumentParser, Namespace
from typing import Dict, List, Set, Tuple

from torch import Tensor, cat, eye, zeros
from torch.nn import Module
from torch.optim import Optimizer

from kfac_pinns_exp.poisson_equation import (
    evaluate_boundary_gramian,
    evaluate_boundary_loss,
    evaluate_interior_gramian,
    evaluate_interior_loss,
)
from kfac_pinns_exp.utils import exponential_moving_average


def parse_GramianOptimizer_args(verbose: bool = False) -> Namespace:
    """Parse command-line arguments for the Gramian optimizer.

    Args:
        verbose: Whether to print the parsed arguments. Default: `False`.

    Returns:
        A namespace with the parsed arguments.
    """
    parser = ArgumentParser(description="Gramian optimizer parameters.")
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate for the Gramian optimizer.",
        required=True,
    )
    parser.add_argument(
        "--damping",
        type=float,
        help="Damping of the Gramian before inversion.",
        required=True,
    )
    parser.add_argument(
        "--ema_factor",
        type=float,
        default=0.95,
        help="Exponential moving average factor for the Gramian.",
    )
    parser.add_argument(
        "--gramian",
        type=str,
        default="full",
        choices=GramianOptimizer.SUPPORTED_GRAMIANS,
        help="Type of Gramian matrix to use.",
    )
    args, _ = parser.parse_known_args()

    if verbose:
        print(f"Gramian optimizer arguments: {args}")

    return args


class GramianOptimizer(Optimizer):
    """Optimizer that uses the exact Gramian matrix."""

    SUPPORTED_GRAMIANS: Set[str] = {"full"}  # , "diagonal", "block_diagonal"]

    def __init__(
        self,
        model: Module,
        lr: float,
        damping: float,
        ema_factor: float = 0.95,
        gramian: str = "full",
    ):
        """Initialize the optimizer.

        Args:
            model: Model to optimize.
            lr: Learning rate.
            damping: Damping of the Gramian before inversion.
            ema_factor: Exponential moving average factor for the Gramian. Default: 0.95.
            verbose: Whether to print the parsed arguments. Default: `False`.
        """
        if gramian not in self.SUPPORTED_GRAMIANS:
            raise ValueError(
                f"Unsupported Gramian type: {gramian}. "
                f"Supported types: {self.SUPPORTED_GRAMIANS}."
            )
        if not 0 <= ema_factor < 1:
            raise ValueError(
                "Exponential moving average factor must be in [0, 1). "
                + f"Got {ema_factor}."
            )
        if lr <= 0.0:
            raise ValueError(f"Learning rate must be positive. Got {lr}.")
        if damping < 0.0:
            raise ValueError(f"Damping factor must be non-negative. Got {damping}.")

        defaults = dict(
            lr=lr,
            damping=damping,
            ema_factor=ema_factor,
            gramian=gramian,
        )
        params = list(model.parameters())
        super().__init__(params, defaults)

        # initialize Gramian and inverse Gramian
        num_params = sum(p.numel() for p in params)
        (dev,) = {p.device for p in params}
        (dt,) = {p.dtype for p in params}
        kwargs = {"device": dev, "dtype": dt}
        initialize_to_identity = True
        self.gramian = (
            eye(num_params, **kwargs)
            if initialize_to_identity
            else zeros(num_params, num_params, **kwargs)
        )
        self.inv_gramian = eye(num_params, **kwargs)
        self.model = model

    def step(
        self, X_Omega: Tensor, y_Omega: Tensor, X_dOmega: Tensor, y_dOmega: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Take a step.

        Args:
            X_Omega: Input for the interior loss.
            y_Omega: Target for the interior loss.
            X_dOmega: Input for the boundary loss.
            y_dOmega: Target for the boundary loss.

        Returns:
            Tuple of the interior and boundary loss before taking the step.
        """
        # compute gradients and Gramians on current data
        interior_gramian = evaluate_interior_gramian(self.model, X_Omega)
        interior_loss = evaluate_interior_loss(self.model, X_Omega, y_Omega)
        interior_loss.backward()

        boundary_gramian = evaluate_boundary_gramian(self.model, X_dOmega)
        boundary_loss = evaluate_boundary_loss(self.model, X_dOmega, y_dOmega)
        boundary_loss.backward()

        # update Gramian
        group = self.param_groups[0]
        ema_factor = group["ema_factor"]
        exponential_moving_average(
            self.gramian, interior_gramian + boundary_gramian, ema_factor
        )

        # update the inverse Gramian
        damping = group["damping"]
        self.inv_gramian = (
            self.gramian + damping * eye(self.gramian.size(0))
        ).inverse()

        # compute natural gradients
        params = group["params"]
        grad_flat = cat([p.grad.flatten() for p in params])
        nat_grad = self.inv_gramian @ grad_flat
        nat_grad = nat_grad.split([p.numel() for p in params])
        nat_grad = [g.reshape(*p.shape) for g, p in zip(nat_grad, params)]

        # update parameters
        lr = group["lr"]
        for param, nat_grad in zip(params, nat_grad):
            param.data.sub_(nat_grad, alpha=lr)

        return interior_loss, boundary_loss
