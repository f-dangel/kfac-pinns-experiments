"""Implements enery-natural gradient descent flavours from Mueller et al. 2023."""

from argparse import ArgumentParser, Namespace
from typing import List, Set, Tuple, Union

from torch import Tensor, cat, eye, logspace, ones, zeros
from torch.linalg import lstsq
from torch.nn import Module
from torch.optim import Optimizer

from kfac_pinns_exp import heat_equation, poisson_equation
from kfac_pinns_exp.autodiff_utils import autograd_gramian
from kfac_pinns_exp.optim.line_search import (
    grid_line_search,
    parse_grid_line_search_args,
)
from kfac_pinns_exp.parse_utils import parse_known_args_and_remove_from_argv
from kfac_pinns_exp.utils import exponential_moving_average


def parse_ENGD_args(verbose: bool = False, prefix: str = "ENGD_") -> Namespace:
    """Parse command-line arguments for the ENGD optimizer.

    Args:
        verbose: Whether to print the parsed arguments. Default: `False`.
        prefix: Prefix for the arguments. Default: "ENGD_".

    Returns:
        A namespace with the parsed arguments.
    """
    parser = ArgumentParser(description="ENGD optimizer parameters.")
    parser.add_argument(
        f"--{prefix}lr",
        help="Learning rate for the Gramian optimizer (float or string).",
        default="grid_line_search",
    )
    parser.add_argument(
        f"--{prefix}ema_factor",
        type=float,
        default=0.0,
        help="Exponential moving average factor for the Gramian.",
    )
    parser.add_argument(
        f"--{prefix}damping",
        type=float,
        default=0.0,
        help="Damping of the Gramian.",
    )
    parser.add_argument(
        f"--{prefix}approximation",
        type=str,
        default="full",
        choices=ENGD.SUPPORTED_APPROXIMATIONS,
        help="Type of Gramian matrix to use.",
    )
    parser.add_argument(
        f"--{prefix}initialize_to_identity",
        action="store_true",
        help="Initialize the Gramian matrix to the identity matrix.",
    )
    parser.add_argument(
        f"--{prefix}equation",
        type=str,
        choices=ENGD.SUPPORTED_EQUATIONS,
        help="The equation to solve.",
        default="poisson",
    )
    args = parse_known_args_and_remove_from_argv(parser)

    lr = f"{prefix}lr"
    if any(char.isdigit() for char in getattr(args, lr)):
        setattr(args, lr, float(getattr(args, lr)))

    if getattr(args, lr) == "grid_line_search":
        # generate the grid from the command line arguments and overwrite the
        # `ENGD_lr` entry with a tuple containing the grid
        grid = parse_grid_line_search_args(verbose=verbose)
        setattr(args, lr, (getattr(args, lr), grid))

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
        SUPPORTED_EQUATIONS: Set of supported PDEs.
    """

    SUPPORTED_APPROXIMATIONS: Set[str] = {"full", "diagonal", "per_layer"}
    SUPPORTED_EQUATIONS: Set[str] = {"poisson", "heat"}

    def __init__(
        self,
        model: Module,
        lr: Union[float, Tuple[str, List[float]]] = ENGD_DEFAULT_LR,
        damping: float = 0.0,
        ema_factor: float = 0.0,
        approximation: str = "full",
        initialize_to_identity: bool = False,
        equation: str = "poisson",
    ):
        """Initialize the ENGD optimizer.

        Args:
            model: Model to optimize.
            lr: Learning rate or tuple specifying the line search strategy.
                Default value is the grid line search used in the paper.
            damping: Damping of the Gramian. Default: `0.0`.
            ema_factor: Factor for the exponential moving average with which previous
                Gramians are accumulated. `0.0` means past Gramians are discarded.
                Default: `0.0`.
            approximation: Type of Gramian matrix to use. Default: `'full'`.
                Other options are `'diagonal'` and `'per_layer'`.
            initialize_to_identity: Whether to initialize the Gramian to the identity
                matrix. Default: `False`. If `True`, the Gramian is initialized to
                identity.
            equation: PDE to solve. Can be `'poisson'` or `'heat'`.
                Default: `'poisson'`.
        """
        self._check_hyperparameters(
            model, lr, damping, ema_factor, approximation, equation
        )
        defaults = dict(
            lr=lr,
            damping=damping,
            ema_factor=ema_factor,
            approximation=approximation,
            initialize_to_identity=initialize_to_identity,
        )
        super().__init__(list(model.parameters()), defaults)

        self.equation = equation
        self.model = model
        self.gramian = self._initialize_curvature()

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
        (
            interior_loss,
            boundary_loss,
        ) = self._eval_loss_and_gradient_and_update_curvature(
            X_Omega, y_Omega, X_dOmega, y_dOmega
        )
        directions = self._compute_natural_gradients()
        self._update_parameters(directions, X_Omega, y_Omega, X_dOmega, y_dOmega)

        return interior_loss, boundary_loss

    @classmethod
    def _check_hyperparameters(
        cls,
        model: Module,
        lr: Union[float, Tuple[str, List[float]]],
        damping: float,
        ema_factor: float,
        approximation: str,
        equation: str,
    ):
        """Verify the supplied constructor arguments.

        Args:
            model: Model to optimize.
            lr: Learning rate or tuple specifying the line search strategy.
            damping: Damping of the Gramian.
            ema_factor: Factor for the exponential moving average with which previous
                Gramians are accumulated.
            approximation: Type of Gramian matrix to use.
            equation: PDE to solve.

        Raises:
            ValueError: If one of the supplied arguments has invalid value.
            NotImplementedError: If the supplied argument combination is unsupported.
        """
        if approximation not in cls.SUPPORTED_APPROXIMATIONS:
            raise ValueError(
                f"Unsupported Gramian type: {approximation}. "
                f"Supported types: {cls.SUPPORTED_APPROXIMATIONS}."
            )
        if damping < 0.0:
            raise ValueError(f"Damping must be non-negative. Got {damping}.")
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
        if not isinstance(model, Module):
            raise ValueError(f"Model must be a torch.nn.Module. Got {type(model)}.")
        if equation not in cls.SUPPORTED_EQUATIONS:
            raise ValueError(
                f"Unsupported equation: {equation}. Supported equations: "
                f"{cls.SUPPORTED_EQUATIONS}."
            )

    def _initialize_curvature(self) -> Union[Tensor, List[Tensor]]:
        """Initialize the Gramian matrices.

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
        identity = self.param_groups[0]["initialize_to_identity"]

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
        elif approximation == "per_layer":
            block_sizes = [
                sum(p.numel() for p in layer.parameters())
                for layer in self.model.modules()
                if list(layer.parameters()) and not list(layer.children())
            ]
            return [
                eye(size, **kwargs) if identity else zeros(size, size, **kwargs)
                for size in block_sizes
            ]
        else:
            raise NotImplementedError(f"Approximation {approximation} not implemented.")

    def _compute_natural_gradients(self) -> List[Tensor]:
        """Compute the natural gradients from current preconditioner and gradients.

        Returns:
            Natural gradients in parameter list format.

        Raises:
            NotImplementedError: If the chosen approximation is not supported.
        """
        params = self.param_groups[0]["params"]
        approximation = self.param_groups[0]["approximation"]
        damping = self.param_groups[0]["damping"]

        # NOTE lstsq only supports the 'gels' driver on CUDA, which assumes the
        # Gramian is full-rank. This assumption is usually violated, hence we
        # off-load the computation to the CPU and use the more stable 'gelsd' driver.
        (original_dev,) = {p.device for p in params}

        grad_flat = cat([p.grad.flatten().cpu() for p in params])

        # compute flattened natural gradient
        if approximation == "full":
            damped_gramian = self.gramian + damping * eye(
                self.gramian.shape[0], device=original_dev, dtype=self.gramian.dtype
            )
            # solve the linear system argmin_x ||Gx + g||_2
            linsolve = lstsq(
                damped_gramian.cpu(), grad_flat.unsqueeze(-1), driver="gelsd"
            )
            nat_grad = -linsolve.solution.to(original_dev).squeeze(-1)
        elif approximation == "diagonal":
            damped_gramian = self.gramian + damping
            # solve `D` 1x1 least-squares problems in parallel
            linsolve = lstsq(
                damped_gramian.cpu().unsqueeze(-1).unsqueeze(-1),
                grad_flat.unsqueeze(-1).unsqueeze(-1),
                driver="gelsd",
            )
            nat_grad = -linsolve.solution.squeeze(-1).squeeze(-1).to(original_dev)
        elif approximation == "per_layer":
            block_sizes = [
                sum(p.numel() for p in layer.parameters())
                for layer in self.model.modules()
                if list(layer.parameters()) and not list(layer.children())
            ]
            nat_grads = []
            for gram, g_flat in zip(self.gramian, grad_flat.split(block_sizes)):
                damped_gram = gram + damping * eye(
                    gram.shape[0], device=original_dev, dtype=gram.dtype
                )
                linsolve = lstsq(
                    damped_gram.cpu(), g_flat.unsqueeze(-1), driver="gelsd"
                )
                ng = -linsolve.solution.to(original_dev).squeeze(-1)
                nat_grads.append(ng)
            nat_grad = cat(nat_grads)
        else:
            raise NotImplementedError(f"Approximation {approximation} not implemented.")

        # un-flatten
        nat_grad = nat_grad.split([p.numel() for p in params])
        return [g.reshape(*p.shape) for g, p in zip(nat_grad, params)]

    def _update_parameters(
        self,
        directions: List[Tensor],
        X_Omega: Tensor,
        y_Omega: Tensor,
        X_dOmega: Tensor,
        y_dOmega: Tensor,
    ):
        """Update the model parameters with the negative natural gradient.

        Args:
            directions: Negative natural gradient in parameter list format.
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
            for param, direction in zip(params, directions):
                param.data.add_(direction, alpha=lr)
        elif lr[0] == "grid_line_search":

            def f() -> Tensor:
                """Closure to evaluate the loss.

                Returns:
                    The loss value.
                """
                interior_loss = self.eval_loss(X_Omega, y_Omega, "interior")
                boundary_loss = self.eval_loss(X_dOmega, y_dOmega, "boundary")
                return interior_loss + boundary_loss

            grid = lr[1]
            grid_line_search(f, params, directions, grid)

        else:
            raise NotImplementedError(f"Line search {lr[0]} not implemented.")

    def _eval_loss_and_gradient_and_update_curvature(
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
        # compute gradients and Gramians on current data
        interior_gramian = self.eval_gramian(X_Omega, "interior")
        interior_loss = self.eval_loss(X_Omega, y_Omega, "interior")
        interior_loss.backward()

        boundary_gramian = self.eval_gramian(X_dOmega, "boundary")
        boundary_loss = self.eval_loss(X_dOmega, y_dOmega, "boundary")
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

    def eval_loss(self, X: Tensor, y: Tensor, loss_type: str) -> Tensor:
        """Evaluate the loss.

        Args:
            X: Input data.
            y: Target data.
            loss_type: Type of the loss function. Can be `'interior'` or `'boundary'`.

        Returns:
            The differentiable loss.
        """
        loss_evaluator = {
            "poisson": {
                "interior": poisson_equation.evaluate_interior_loss,
                "boundary": poisson_equation.evaluate_boundary_loss,
            },
            "heat": {
                "interior": heat_equation.evaluate_interior_loss,
                "boundary": heat_equation.evaluate_boundary_loss,
            },
        }[self.equation][loss_type]
        loss, _, _ = loss_evaluator(self.model, X, y)
        return loss

    def eval_gramian(self, X: Tensor, loss_type: str) -> Union[Tensor, List[Tensor]]:
        """Evaluate the Gramian matrix.

        Args:
            X: Batched input data on which the Gramian is evaluated.
            loss_type: Type of the loss function. Can be `'interior'` or `'boundary'`.

        Returns:
            The Gramian matrix or a list of Gramian matrices, depending on the
            approximation that is specified.

        Raises:
            ValueError: If the approximation is not one of `'full'`, `'diagonal'`, or
                `'per_layer'`.
        """
        approximation = self.param_groups[0]["approximation"]
        batch_size = X.shape[0]
        param_names = [n for n, _ in self.model.named_parameters()]
        gramian = autograd_gramian(
            self.model,
            X,
            param_names,
            loss_type=f"{self.equation}_{loss_type}",
            approximation=approximation,
        )
        if approximation == "per_layer":
            return [g.div_(batch_size) for g in gramian]
        elif approximation in {"full", "diagonal"}:
            return gramian.div_(batch_size)
        else:
            raise ValueError(
                f"Unknown approximation {approximation!r}. "
                "Must be one of 'full', 'diagonal', or 'per_layer'."
            )
