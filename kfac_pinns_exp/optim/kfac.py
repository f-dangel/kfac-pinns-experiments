"""Implements the KFAC-for-PINNs optimizer."""

from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Tuple, Union

from torch import Tensor, cat, dtype, eye, float64
from torch.nn import Module
from torch.optim import Optimizer

from kfac_pinns_exp import heat_equation, poisson_equation
from kfac_pinns_exp.inverse_kronecker_sum import InverseKroneckerSum
from kfac_pinns_exp.kfac_utils import check_layers_and_initialize_kfac
from kfac_pinns_exp.optim.engd import ENGD_DEFAULT_LR
from kfac_pinns_exp.optim.line_search import (
    backtracking_line_search,
    grid_line_search,
    parse_backtracking_line_search_args,
    parse_grid_line_search_args,
)
from kfac_pinns_exp.parse_utils import parse_known_args_and_remove_from_argv
from kfac_pinns_exp.utils import exponential_moving_average


def parse_KFAC_args(verbose: bool = False, prefix="KFAC_") -> Namespace:
    """Parse command-line arguments for `KFAC`.

    Args:
        verbose: Whether to print the parsed arguments. Default: `False`.
        prefix: The prefix for the arguments. Default: `'KFAC_'`.

    Returns:
        A namespace with the parsed arguments.
    """
    DTYPES = {"float64": float64}
    parser = ArgumentParser(description="Parse arguments for setting up KFAC.")

    parser.add_argument(
        f"--{prefix}lr",
        help="Learning rate or line search strategy for the optimizer.",
        default="grid_line_search",
    )
    parser.add_argument(
        f"--{prefix}damping",
        type=float,
        help="Damping factor for the optimizer.",
        required=True,
    )
    parser.add_argument(
        f"--{prefix}T_kfac",
        type=int,
        help="Update frequency of KFAC matrices.",
        default=1,
    )
    parser.add_argument(
        f"--{prefix}T_inv",
        type=int,
        help="Update frequency of the inverse KFAC matrices.",
        default=1,
    )
    parser.add_argument(
        f"--{prefix}ema_factor",
        type=float,
        help="Exponential moving average factor for the KFAC matrices.",
        default=0.95,
    )
    parser.add_argument(
        f"--{prefix}ggn_type",
        type=str,
        choices=KFAC.SUPPORTED_GGN_TYPES,
        help="Determines type of backpropagated error used to compute KFAC.",
        default="type-2",
    )
    parser.add_argument(
        f"--{prefix}kfac_approx",
        type=str,
        choices=KFAC.SUPPORTED_KFAC_APPROXIMATIONS,
        help="Approximation method for the KFAC matrices.",
        default="expand",
    )
    parser.add_argument(
        f"--{prefix}inv_strategy",
        type=str,
        choices=["invert kronecker sum"],
        help="Inversion strategy for KFAC.",
        default="invert kronecker sum",
    )
    parser.add_argument(
        f"--{prefix}inv_dtype",
        type=str,
        choices=DTYPES.keys(),
        help="Data type for the inverse KFAC matrices.",
        default="float64",
    )
    parser.add_argument(
        f"--{prefix}initialize_to_identity",
        action="store_true",
        help="Whether to initialize the KFAC matrices to identity.",
    )
    parser.add_argument(
        f"--{prefix}equation",
        type=str,
        choices=KFAC.SUPPORTED_EQUATIONS,
        help="The equation to solve.",
        default="poisson",
    )
    parser.add_argument(
        f"--{prefix}momentum",
        type=float,
        help="Momentum on the update.",
        default=0.0,
    )

    args = parse_known_args_and_remove_from_argv(parser)
    # overwrite inv_dtype with value from dictionary
    inv_dtype = f"{prefix}inv_dtype"
    setattr(args, inv_dtype, DTYPES[getattr(args, inv_dtype)])

    # overwrite the lr value
    lr = f"{prefix}lr"
    if any(char.isdigit() for char in getattr(args, lr)):
        setattr(args, lr, float(getattr(args, lr)))

    if getattr(args, lr) == "grid_line_search":
        # generate the grid from the command line arguments and overwrite the
        # `lr` entry with a tuple containing the grid
        grid = parse_grid_line_search_args(verbose=verbose)
        setattr(args, lr, (getattr(args, lr), grid))
    elif getattr(args, lr) == "backtracking_line_search":
        kwargs = parse_backtracking_line_search_args(verbose=verbose)
        setattr(args, lr, (getattr(args, lr), kwargs))

    if verbose:
        print("Parsed arguments for KFAC: ", args)

    return args


class KFAC(Optimizer):
    """KFAC optimizer for PINN problems.

    Attributes:
        SUPPORTED_KFAC_APPROXIMATIONS: Available KFAC approximations. Supports
            KFAC-expand and KFAC-reduce.
        SUPPORTED_GGN_TYPES: Available approximations of the GGN used to approximate
            KFAC. Currently supports `'type-2'`, `'empirical'`, and `'forward-only'`
            (ordered in descending computational cost and approximation quality).
        SUPPORTED_EQUATIONS: Available equations to solve. Currently supports the
            Poisson (`'poisson'`) and heat (`'heat'`) equations.
    """

    SUPPORTED_KFAC_APPROXIMATIONS = {"expand", "reduce"}
    SUPPORTED_GGN_TYPES = {"type-2", "empirical", "forward-only"}
    SUPPORTED_EQUATIONS = {"poisson", "heat"}

    def __init__(
        self,
        layers: List[Module],
        damping: float,
        lr: Union[
            float, Tuple[str, Union[List[float], Dict[str, Any]]]
        ] = ENGD_DEFAULT_LR,
        T_kfac: int = 1,
        T_inv: int = 1,
        ema_factor: float = 0.95,
        kfac_approx: str = "expand",
        inv_strategy: str = "invert kronecker sum",
        ggn_type: str = "type-2",
        inv_dtype: dtype = float64,
        initialize_to_identity: bool = False,
        equation: str = "poisson",
        momentum: float = 0.0,
    ) -> None:
        """Set up the optimizer.

        Limitations:
            - No parameter group support. Can only train all parameters.

        Args:
            layers: List of layers of the neural network.
            damping: Damping factor. Must be positive.
            lr: Positive learning rate or tuple specifying the line search. By default
                uses the same grid line search as the ENGD optimizer. Also supports
                using a backtracking line search.
            T_kfac: Positive integer specifying the update frequency for
                the boundary and the interior terms' KFACs. Default is `1`.
            T_inv: Positive integer specifying the pre-conditioner update
                frequency. Default is `1`.
            ema_factor: Exponential moving average factor for the KFAC factors. Must be
                in `[0, 1)`. Default is `0.95`.
            kfac_approx: KFAC approximation method. Must be either `'expand'`, or
                `'reduce'`. Defaults to `'expand'`.
            ggn_type: Type of the GGN to use. This influences the backpropagted error
                used to compute the KFAC matrices. Can be either `'type-2'`,
                `'empirical'`, or `'forward-only'`. Default: `'type-2'`.
            inv_strategy: Inversion strategy. Must `'invert kronecker sum'`. Default is
                `'invert kronecker sum'`.
            inv_dtype: Data type to carry out the curvature inversion. Default is
                `torch.float64`. The pre-conditioner will be converted back to the same
                data type as the parameters after the inversion.
            initialize_to_identity: Whether to initialize the KFAC factors to the
                identity matrix. Default is `False` (initialize with zero).
            equation: Equation to solve. Currently supports `'poisson'` and `'heat'`.
                Default: `'poisson'`.
            momentum: Momentum on the update. Default: `0.0`.

        Raises:
            ValueError: If the supplied equation is unsupported.
        """
        defaults = dict(
            lr=lr,
            damping=damping,
            T_kfac=T_kfac,
            T_inv=T_inv,
            ema_factor=ema_factor,
            kfac_approx=kfac_approx,
            ggn_type=ggn_type,
            inv_strategy=inv_strategy,
            inv_dtype=inv_dtype,
            initialize_to_identity=initialize_to_identity,
            momentum=momentum,
        )
        params = sum((list(layer.parameters()) for layer in layers), [])
        super().__init__(params, defaults)
        self._check_hyperparameters()

        if equation not in self.SUPPORTED_EQUATIONS:
            raise ValueError(
                f"Equation {equation} not supported."
                f" Supported are: {self.SUPPORTED_EQUATIONS}."
            )
        self.equation = equation

        # initialize KFAC matrices for the interior and boundary term
        self.kfacs_interior = check_layers_and_initialize_kfac(
            layers, initialize_to_identity=initialize_to_identity
        )
        self.kfacs_boundary = check_layers_and_initialize_kfac(
            layers, initialize_to_identity=initialize_to_identity
        )

        self.steps = 0
        self.inv: Dict[int, Union[InverseKroneckerSum, Tensor]] = {}
        self.layers = layers
        self.layer_idxs = [
            idx for idx, layer in enumerate(self.layers) if list(layer.parameters())
        ]

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
        loss_interior = self.eval_loss_and_update_kfac(X_Omega, y_Omega, "interior")
        loss_interior.backward()
        loss_boundary = self.eval_loss_and_update_kfac(X_dOmega, y_dOmega, "boundary")
        loss_boundary.backward()

        self.update_preconditioner()

        directions = []
        for layer_idx in self.layer_idxs:
            nat_grad_weight, nat_grad_bias = self.compute_natural_gradient(layer_idx)
            directions.extend([-nat_grad_weight, -nat_grad_bias])
        self.add_momentum(directions)

        self._update_parameters(directions, X_Omega, y_Omega, X_dOmega, y_dOmega)

        self.steps += 1

        return loss_interior, loss_boundary

    def update_preconditioner(self) -> None:
        """Update the inverse damped KFAC."""
        group = self.param_groups[0]
        T_inv = group["T_inv"]

        if self.steps % T_inv != 0:
            return

        inv_dtype = group["inv_dtype"]
        damping = group["damping"]

        # compute the KFAC inverse
        for layer_idx in self.layer_idxs:
            weight_dtype = self.layers[layer_idx].weight.dtype
            weight_device = self.layers[layer_idx].weight.device

            # NOTE that in the literature (column-stacking), KFAC w.r.t. the flattened
            # weights is A₁ ⊗ A₂ + B₁ ⊗ B₂. However, in code we use row-stacking
            # flattening. Effectively, we have to swap the Kronecker factors to obtain
            # KFAC w.r.t. the flattened (row-stacking) weights.
            A2, A1 = self.kfacs_interior[layer_idx]
            B2, B1 = self.kfacs_boundary[layer_idx]

            # add the damping
            kwargs = {"dtype": weight_dtype, "device": weight_device}
            A1 = A1 + damping * eye(*A1.shape, **kwargs)
            A2 = A2 + damping * eye(*A2.shape, **kwargs)
            B1 = B1 + damping * eye(*B1.shape, **kwargs)
            B2 = B2 + damping * eye(*B2.shape, **kwargs)

            self.inv[layer_idx] = InverseKroneckerSum(  # noqa: B909
                A1, A2, B1, B2, inv_dtype=inv_dtype
            )

    def compute_natural_gradient(self, layer_idx: int) -> Tuple[Tensor, Tensor]:
        """Compute the natural gradient for the specified layer.

        Args:
            layer_idx: Index of the layer for which the natural gradient is computed.

        Returns:
            Tuple of natural gradients for the weight and bias.
        """
        layer = self.layers[layer_idx]
        grad_combined = cat(
            [layer.weight.grad.data, layer.bias.grad.data.unsqueeze(-1)], dim=1
        )
        _, d_in = layer.weight.shape
        nat_grad_combined = self.inv[layer_idx] @ grad_combined
        nat_grad_weight, nat_grad_bias = nat_grad_combined.split([d_in, 1], dim=1)
        return nat_grad_weight, nat_grad_bias.squeeze(1)

    def _check_hyperparameters(self):  # noqa: C901
        """Check the hyperparameters for the KFAC optimizer.

        Raises:
            ValueError: If any hyperparameter is invalid.
        """
        num_groups = len(self.param_groups)
        if num_groups != 1:
            raise ValueError(
                f"KFAC optimizer expects exactly 1 parameter group. Got {num_groups}."
            )
        (group,) = self.param_groups

        T_kfac = group["T_kfac"]
        if T_kfac <= 0:
            raise ValueError(f"T_kfac must be positive. Got {T_kfac}.")

        T_inv = group["T_inv"]
        if T_inv <= 0:
            raise ValueError(f"T_inv must be positive. Got {T_inv}.")

        kfac_approx = group["kfac_approx"]
        if kfac_approx not in self.SUPPORTED_KFAC_APPROXIMATIONS:
            raise ValueError(
                f"Unsupported KFAC approximation: {kfac_approx}. "
                + f"Supported: {self.SUPPORTED_KFAC_APPROXIMATIONS}."
            )

        ggn_type = group["ggn_type"]
        if ggn_type not in self.SUPPORTED_GGN_TYPES:
            raise ValueError(
                f"Unsupported GGN type: {ggn_type}. "
                + f"Supported: {self.SUPPORTED_GGN_TYPES}."
            )

        ema_factor = group["ema_factor"]
        if not 0 <= ema_factor < 1:
            raise ValueError(
                "Exponential moving average factor must be in [0, 1). "
                f"Got {ema_factor}."
            )

        lr = group["lr"]
        if isinstance(lr, float):
            if lr <= 0.0:
                raise ValueError(f"Learning rate must be positive. Got {lr}.")
        elif lr[0] not in {"grid_line_search", "backtracking_line_search"}:
            raise ValueError(f"Unsupported line search: {lr[0]}.")

        damping = group["damping"]
        if damping < 0.0:
            raise ValueError(f"Damping factor must be non-negative. Got {damping}.")

        inv_strategy = group["inv_strategy"]
        if inv_strategy != "invert kronecker sum":
            raise ValueError(f"Unsupported inversion strategy: {inv_strategy}.")

        momentum = group["momentum"]
        if not 0 <= momentum < 1:
            raise ValueError(f"Momentum must be in the range [0, 1). Got {momentum}.")

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
            ValueError: If the chosen line search is not supported.
        """
        lr = self.param_groups[0]["lr"]
        params = self.param_groups[0]["params"]

        if isinstance(lr, float):
            for param, direction in zip(params, directions):
                param.data.add_(direction, alpha=lr)
        else:

            def f() -> Tensor:
                """Closure to evaluate the loss.

                Returns:
                    Loss value.
                """
                interior_loss = self.eval_loss(X_Omega, y_Omega, "interior")
                boundary_loss = self.eval_loss(X_dOmega, y_dOmega, "boundary")
                return interior_loss + boundary_loss

            if lr[0] == "grid_line_search":
                grid = lr[1]
                grid_line_search(f, params, directions, grid)
            elif lr[0] == "backtracking_line_search":
                kwargs = lr[1]
                backtracking_line_search(f, params, directions, **kwargs)
            else:
                raise ValueError(f"Unsupported line search: {lr[0]}.")

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
        loss, _, _ = loss_evaluator(self.layers, X, y)
        return loss

    def eval_loss_and_update_kfac(self, X: Tensor, y: Tensor, loss_type: str) -> Tensor:
        """Evaluate the loss, update the KFAC factors, and return the loss.

        Args:
            X: Boundary input data.
            y: Boundary label data.
            loss_type: Type of the loss function. Can be `'interior'` or `'boundary'`.

        Returns:
            Differentiable loss.
        """
        group = self.param_groups[0]

        if self.steps % group["T_kfac"] != 0:
            return self.eval_loss(X, y, loss_type)

        # compute loss and KFAC matrices
        ggn_type = group["ggn_type"]
        kfac_approx = group["kfac_approx"]
        loss_and_kfac_evaluator = {
            "poisson": {
                "interior": poisson_equation.evaluate_interior_loss_and_kfac,
                "boundary": poisson_equation.evaluate_boundary_loss_and_kfac,
            },
            "heat": {
                "interior": heat_equation.evaluate_interior_loss_and_kfac,
                "boundary": heat_equation.evaluate_boundary_loss_and_kfac,
            },
        }[self.equation][loss_type]
        loss, kfacs = loss_and_kfac_evaluator(
            self.layers, X, y, ggn_type=ggn_type, kfac_approx=kfac_approx
        )

        # update KFAC matrices
        ema_factor = group["ema_factor"]
        for layer_idx in self.layer_idxs:
            destinations = {
                "boundary": self.kfacs_boundary,
                "interior": self.kfacs_interior,
            }[loss_type][layer_idx]
            updates = kfacs[layer_idx]
            for destination, update in zip(destinations, updates):
                exponential_moving_average(destination, update, ema_factor)

        return loss

    def add_momentum(self, directions: List[Tensor]):
        """Incorporate momentum into the update direction (in-place).

        Args:
            directions: Update directions in list format.
        """
        group = self.param_groups[0]
        momentum = group["momentum"]
        if momentum == 0.0:
            return

        for d, p in zip(directions, group["params"]):
            if self.steps == 0:  # initialize momentum buffers
                self.state[p]["momentum_buffer"] = d
            else:  # update internal momentum buffer and direction
                p_mom = self.state[p]["momentum_buffer"]
                p_mom.mul_(momentum).add_(d)
                d.copy_(p_mom)
