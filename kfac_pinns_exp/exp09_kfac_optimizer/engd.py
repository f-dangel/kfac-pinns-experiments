"""Implement enery-natural gradient descent flavours from Mueller et al. 2023."""

from argparse import ArgumentParser, Namespace
from typing import List, Set, Tuple, Union

from torch import Tensor, cat, eye, logspace, ones, zeros
from torch.nn import Module
from torch.optim import Optimizer

from kfac_pinns_exp.exp09_kfac_optimizer.line_search import (
    grid_line_search,
    parse_grid_line_search_args,
)
from kfac_pinns_exp.poisson_equation import (
    evaluate_boundary_gramian,
    evaluate_boundary_loss,
    evaluate_interior_gramian,
    evaluate_interior_loss,
)
from kfac_pinns_exp.utils import exponential_moving_average


def parse_ENGD_args(verbose: bool = False) -> Namespace:
    """Parse command-line arguments for the ENGD optimizer.

    Args:
        verbose: Whether to print the parsed arguments. Default: `False`.

    Returns:
        A namespace with the parsed arguments.
    """
    parser = ArgumentParser(description="ENGD optimizer parameters.")
    parser.add_argument(
        "--ENGD_lr",
        type=float,
        help="Learning rate for the Gramian optimizer.",
        required=True,
    )
    parser.add_argument(
        "--ENGD_damping",
        type=float,
        help="Damping of the Gramian before inversion.",
        required=True,
    )
    parser.add_argument(
        "--ENGD_ema_factor",
        type=float,
        default=0.95,
        help="Exponential moving average factor for the Gramian.",
    )
    parser.add_argument(
        "--ENGD_approximation",
        type=str,
        default="full",
        choices=ENGD.SUPPORTED_APPROXIMATIONS,
        help="Type of Gramian matrix to use.",
    )
    args, _ = parser.parse_known_args()

    if args.ENGD_lr == "grid_line_search":
        # generate the grid from the command line arguments and overwrite the
        # `ENGD_lr` entry with a tuple containing the grid
        grid = parse_grid_line_search_args()
        args.ENGD_lr = (args.ENGD_lr, grid)

    if verbose:
        print(f"ENGD optimizer arguments: {args}")

    return args


# The default learning rate strategy for ENGD is using a line search which evaluates
# the loss on a logarithmic grid and picks the best value.
ENGD_DEFAULT_LR = (
    "grid_line_search",
    logspace(-30, 0, steps=31, base=2).tolist(),
)


class ENGD(Optimizer):
    """Energy natural gradient descent with the exact Gramian matrix.

    Mueller & Zeinhofer, 2023: Achieving high accuracy with Pinns via energy natural
    gradient descent (ICML).

    JAX implementation:
    https://github.com/MariusZeinhofer/Natural-Gradient-PINNs-ICML23/tree/main

    Attributes:
        SUPPORTED_APPROXIMATIONS: Set of supported Gramian approximations.
    """

    SUPPORTED_APPROXIMATIONS: Set[str] = {"full", "diagonal", "per_layer"}

    def __init__(
        self,
        model: Module,
        lr: Union[float, Tuple[str, List[float]]] = ENGD_DEFAULT_LR,
        damping: float = 0.0,
        ema_factor: float = 0.0,
        approximation: str = "full",
    ):
        """Initialize the ENGD optimizer.

        Args:
            model: Model to optimize.
            lr: Learning rate or tuple specifying the line search strategy.
                Default value is the grid line search used in the paper.
            damping: Damping term added to the Gramian before inversion.
            ema_factor: Factor for the exponential moving average with which previous
                Gramians are accumulated. `0.0` means past Gramians are discarded.
                Default: `0.0`.
            approximation: Type of Gramian matrix to use. Default: `'full'`.
                Other options are `'diagonal'` and `'per_layer'`.
        """
        self._check_hyperparameters(model, lr, damping, ema_factor, approximation)
        defaults = dict(
            lr=lr, damping=damping, ema_factor=ema_factor, approximation=approximation
        )
        super().__init__(list(model.parameters()), defaults)

        self.gramian = self._initialize_curvature(identity=True)
        self.inv_gramian = self._initialize_preconditioner()
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
        interior_loss, boundary_loss = (
            self._evalute_loss_and_gradient_and_update_curvature(
                X_Omega, y_Omega, X_dOmega, y_dOmega
            )
        )
        self._update_preconditioner()
        nat_grad = self._compute_natural_gradients()
        self._update_parameters(nat_grad, X_Omega, y_Omega, X_dOmega, y_dOmega)

        return interior_loss, boundary_loss

    @classmethod
    def _check_hyperparameters(
        cls,
        model: Module,
        lr: Union[float, Tuple[str, List[float]]],
        damping: float,
        ema_factor: float,
        approximation: str,
    ):
        """Verify the supplied constructor arguments.

        Args:
            model: Model to optimize.
            lr: Learning rate or tuple specifying the line search strategy.
            damping: Damping term added to the Gramian before inversion.
            ema_factor: Factor for the exponential moving average with which previous
                Gramians are accumulated.
            approximation: Type of Gramian matrix to use.

        Raises:
            ValueError: If one of the supplied arguments has invalid value.
            NotImplementedError: If the supplied argument combination is unsupported.
        """
        if approximation not in cls.SUPPORTED_APPROXIMATIONS:
            raise ValueError(
                f"Unsupported Gramian type: {approximation}. "
                f"Supported types: {cls.SUPPORTED_APPROXIMATIONS}."
            )
        elif approximation == "per_layer":
            raise NotImplementedError(
                f"Approximation {approximation} not yet supported."
            )
        if not 0 <= ema_factor < 1:
            raise ValueError(
                "Exponential moving average factor must be in [0, 1). "
                + f"Got {ema_factor}."
            )
        if isinstance(lr, float):
            if lr <= 0.0:
                raise ValueError(f"Learning rate must be positive. Got {lr}.")
        elif lr[0] != "grid_line_search":
            raise NotImplementedError(f"Line search {lr[0]} not implemented.")
        if damping < 0.0:
            raise ValueError(f"Damping factor must be non-negative. Got {damping}.")
        if not isinstance(model, Module):
            raise ValueError(f"Model must be a torch.nn.Module. Got {type(model)}.")

    def _initialize_curvature(
        self, identity: bool = True
    ) -> Union[Tensor, List[Tensor]]:
        """Initialize the Gramian matrices.

        Args:
            identity: Whether to initialize the Gramian as the identity matrix.
                Otherwise, use zero initialization. Default: `True`.

        Returns:
            The initialized Gramian matrix or a list of Gramian matrices, depending
            on the chosen approximation.

        Raises:
            NotImplementedError: If the chosen approximation is not supported.
        """
        params = self.param_groups[0]["params"]
        num_params = sum(p.numel() for p in params)
        (dev,) = {p.device for p in params}
        (dt,) = {p.dtype for p in params}
        kwargs = {"device": dev, "dtype": dt}

        approximation = self.param_groups[0]["approximation"]

        if approximation == "full":
            return (
                eye(num_params, **kwargs)
                if identity
                else zeros(num_params, num_params, **kwargs)
            )
        elif approximation == "diagonal":
            return (
                ones(num_params, **kwargs) if identity else zeros(num_params, **kwargs)
            )
        else:
            raise NotImplementedError(
                f"Curvature initialization for {approximation} not implemented."
            )

    def _initialize_preconditioner(self) -> Union[Tensor, List[Tensor]]:
        """Initialize the inverse Gramian, i.e. the pre-conditioner.

        Returns:
            The initialized pre-conditioner matrix or a list of pre-conditioner
            matrices, depending on the chosen approximation.

        Raises:
            NotImplementedError: If the chosen approximation is not supported.
        """
        params = self.param_groups[0]["params"]
        approximation = self.param_groups[0]["approximation"]

        num_params = sum(p.numel() for p in params)
        (dev,) = {p.device for p in params}
        (dt,) = {p.dtype for p in params}
        kwargs = {"device": dev, "dtype": dt}

        if approximation == "full":
            return eye(num_params, **kwargs)
        elif approximation == "diagonal":
            return ones(num_params, **kwargs)
        else:
            raise NotImplementedError(
                f"Pre-conditioner init for {approximation} not implemented."
            )

    def _compute_natural_gradients(self) -> List[Tensor]:
        """Compute the natural gradients from current pre-conditioner and gradients.

        Returns:
            Natural gradients in parameter list format.

        Raises:
            NotImplementedError: If the chosen approximation is not supported.
        """
        params = self.param_groups[0]["params"]
        approximation = self.param_groups[0]["approximation"]
        grad_flat = cat([p.grad.flatten() for p in params])

        # compute flattened natural gradient
        if approximation == "full":
            nat_grad = self.inv_gramian @ grad_flat
        elif approximation == "diagonal":
            nat_grad = self.inv_gramian * grad_flat
        else:
            raise NotImplementedError(
                f"Natural gradient computation not implemented for {approximation}."
            )

        # un-flatten
        nat_grad = nat_grad.split([p.numel() for p in params])
        return [g.reshape(*p.shape) for g, p in zip(nat_grad, params)]

    def _update_preconditioner(self):
        """Update the inverse Gramian.

        Raises:
            NotImplementedError: If the chosen approximation is not supported.
        """
        approximation = self.param_groups[0]["approximation"]
        damping = self.param_groups[0]["damping"]

        # update the inverse Gramian
        if approximation == "full":
            kwargs = {"device": self.gramian.device, "dtype": self.gramian.dtype}
            self.inv_gramian = (
                self.gramian + damping * eye(self.gramian.shape[0], **kwargs)
            ).inverse()
        elif approximation == "diagonal":
            self.inv_gramian = 1.0 / (self.gramian + damping)
        else:
            raise NotImplementedError(
                f"Pre-conditioner update not implemented for {approximation}."
            )

    def _update_parameters(
        self,
        directions: List[Tensor],
        X_Omega: Tensor,
        y_Omega: Tensor,
        X_dOmega: Tensor,
        y_dOmega: Tensor,
    ):
        """Update the model parameters with the natural gradient.

        Args:
            directions: Natural gradient in parameter list format.
            X_Omega: Input data on the interior.
            y_Omega: Target data on the interior.
            X_dOmega: Input data on the boundary.
            y_dOmega: Target data on the boundary.

        Raises:
            NotImplementedError: If the chosen line search is not supported.
        """
        lr = self.param_groups[0]["lr"]
        params = self.param_groups[0]["params"]

        if isinstance(lr, float):
            params = self.param_groups[0]["params"]
            for param, direction in zip(params, directions):
                param.data.sub_(direction, alpha=lr)
        elif lr[0] == "grid_line_search":

            def f() -> Tensor:
                """Closure to evaluate the loss.

                Returns:
                    The loss value.
                """
                interior_loss = evaluate_interior_loss(self.model, X_Omega, y_Omega)
                boundary_loss = evaluate_boundary_loss(self.model, X_dOmega, y_dOmega)
                return interior_loss + boundary_loss

            params = self.param_groups[0]["params"]
            grid = lr[1]
            grid_line_search(f, params, directions, grid)

        else:
            raise NotImplementedError(f"Line search {lr[0]} not implemented.")

    def _evaluate_loss_and_gradient_and_update_curvature(
        self, X_Omega: Tensor, y_Omega: Tensor, X_dOmega: Tensor, y_dOmega: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Evaluate the loss and gradient and update the Gramian matrices.

        Gradients are accumulated into `.grad` fields of the parameters.

        Args:
            X_Omega: Input data on the interior.
            y_Omega: Target data on the interior.
            X_dOmega: Input data on the boundary.
            y_dOmega: Target data on the boundary.

        Returns:
            Tuple of the interior and boundary loss.

        Raises:
            NotImplementedError: If the chosen approximation is not supported.
        """
        approximation = self.param_groups[0]["approximation"]
        # compute gradients and Gramians on current data
        interior_gramian = evaluate_interior_gramian(self.model, X_Omega, approximation)
        interior_loss = evaluate_interior_loss(self.model, X_Omega, y_Omega)
        interior_loss.backward()

        boundary_gramian = evaluate_boundary_gramian(
            self.model, X_dOmega, approximation
        )
        boundary_loss = evaluate_boundary_loss(self.model, X_dOmega, y_dOmega)
        boundary_loss.backward()

        ema_factor = self.param_groups[0]["ema_factor"]
        approximation = self.param_groups[0]["approximation"]

        if approximation == "per_layer":
            for gram, int_gram, bound_gram in zip(
                self.gramian, interior_gramian, boundary_gramian
            ):
                exponential_moving_average(gram, int_gram + bound_gram, ema_factor)
        elif approximation in ["diagonal", "full"]:
            exponential_moving_average(
                self.gramian, interior_gramian + boundary_gramian, ema_factor
            )
        else:
            raise NotImplementedError(
                f"Curvature update not implemented for {approximation}."
            )

        return interior_loss, boundary_loss
