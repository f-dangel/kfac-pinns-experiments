"""Implements the KFAC-for-PINNs optimizer."""

from argparse import ArgumentParser, Namespace
from typing import Dict, List, Tuple, Union

from torch import Tensor, cat, dtype, eye, float64
from torch.nn import Module
from torch.optim import Optimizer

from kfac_pinns_exp.exp07_inverse_kronecker_sum.inverse_kronecker_sum import (
    InverseKroneckerSum,
)
from kfac_pinns_exp.exp09_kfac_optimizer.engd import ENGD_DEFAULT_LR
from kfac_pinns_exp.exp09_kfac_optimizer.line_search import (
    grid_line_search,
    parse_grid_line_search_args,
)
from kfac_pinns_exp.exp09_kfac_optimizer.utils import (
    parse_known_args_and_remove_from_argv,
)
from kfac_pinns_exp.kfac_utils import check_layers_and_initialize_kfac
from kfac_pinns_exp.poisson_equation import (
    evaluate_boundary_loss,
    evaluate_boundary_loss_and_kfac_expand,
    evaluate_interior_loss,
    evaluate_interior_loss_and_kfac_expand,
)
from kfac_pinns_exp.utils import exponential_moving_average


def parse_KFAC_args(verbose: bool = False) -> Namespace:
    """Parse command-line arguments for `KFAC`.

    Args:
        verbose: Whether to print the parsed arguments. Default: `False`.

    Returns:
        A namespace with the parsed arguments.
    """
    DTYPES = {"float64": float64}
    parser = ArgumentParser(description="Parse arguments for setting up KFAC.")

    parser.add_argument(
        "--KFAC_lr",
        help="Learning rate or line search strategy for the optimizer.",
        default="grid_line_search",
    )
    parser.add_argument(
        "--KFAC_damping",
        type=float,
        help="Damping factor for the optimizer.",
        required=True,
    )
    parser.add_argument(
        "--KFAC_T_kfac", type=int, help="Update frequency of KFAC matrices.", default=1
    )
    parser.add_argument(
        "--KFAC_T_inv",
        type=int,
        help="Update frequency of the inverse KFAC matrices.",
        default=1,
    )
    parser.add_argument(
        "--KFAC_ema_factor",
        type=float,
        help="Exponential moving average factor for the KFAC matrices.",
        default=0.95,
    )
    parser.add_argument(
        "--KFAC_kfac_approx",
        type=str,
        choices=KFAC.SUPPORTED_KFAC_APPROXIMATIONS,
        help="Approximation method for the KFAC matrices.",
        default="expand",
    )
    parser.add_argument(
        "--KFAC_inv_strategy",
        type=str,
        choices=["invert kronecker sum"],
        help="Inversion strategy for KFAC.",
        default="invert kronecker sum",
    )
    parser.add_argument(
        "--KFAC_inv_dtype",
        type=str,
        choices=DTYPES.keys(),
        help="Data type for the inverse KFAC matrices.",
        default="float64",
    )
    parser.add_argument(
        "--KFAC_initialize_to_identity",
        action="store_true",
        help="Whether to initialize the KFAC matrices to identity.",
    )
    args = parse_known_args_and_remove_from_argv(parser)
    # overwrite inv_dtype with value from dictionary
    args.KFAC_inv_dtype = DTYPES[args.KFAC_inv_dtype]

    # overwrite the lr value
    if any(char.isdigit() for char in args.KFAC_lr):
        args.KFAC_lr = float(args.KFAC_lr)

    if args.KFAC_lr == "grid_line_search":
        # generate the grid from the command line arguments and overwrite the
        # `KFAC_lr` entry with a tuple containing the grid
        grid = parse_grid_line_search_args()
        args.KFAC_lr = (args.KFAC_lr, grid)

    if verbose:
        print("Parsed arguments for KFAC: ", args)

    return args


class KFAC(Optimizer):
    """KFAC optimizer for PINN problems."""

    SUPPORTED_KFAC_APPROXIMATIONS = {"expand"}

    def __init__(
        self,
        layers: List[Module],
        damping: float,
        lr: Union[float, Tuple[str, List[float]]] = ENGD_DEFAULT_LR,
        T_kfac: int = 1,
        T_inv: int = 1,
        ema_factor: float = 0.95,
        kfac_approx: str = "expand",
        inv_strategy: str = "invert kronecker sum",
        inv_dtype: dtype = float64,
        initialize_to_identity: bool = False,
    ) -> None:
        """Set up the optimizer.

        Limitations:
            - No parameter group support. Can only train all parameters.
            - No support for KFAC-reduce.
            - No support for input-based curvature only.
            - No performance gains if `T_kfac` is increased as the KFAC factors are
              always computed and either incorporated or discarded.

        Args:
            layers: List of layers of the neural network.
            damping: Damping factor. Must be positive.
            lr: Positive learning rate or tuple specifying the line search. By default
                uses the same line search as the ENGD optimizer.
            T_kfac: Positive integer specifying the update frequency for
                the boundary and the interior terms' KFACs. Default is `1`.
            T_inv: Positive integer specifying the pre-conditioner update
                frequency. Default is `1`.
            ema_factor: Exponential moving average factor for the KFAC factors. Must be
                in `[0, 1)`. Default is `0.95`.
            kfac_approx: KFAC approximation method. Must be either `'expand'`, or
                `'reduce'`. Defaults to `'expand'`.
            inv_strategy: Inversion strategy. Must `'invert kronecker sum'`. Default is
                `'invert kronecker sum'`.
            inv_dtype: Data type to carry out the curvature inversion. Default is
                `torch.float64`. The pre-conditioner will be converted back to the same
                data type as the parameters after the inversion.
            initialize_to_identity: Whether to initialize the KFAC factors to the
                identity matrix. Default is `False` (initialize with zero).

        Raises:
            ValueError: If any of the hyper-parameters is invalid.
        """
        self._check_hyperparameters(
            lr,
            damping,
            T_kfac,
            T_inv,
            ema_factor,
            kfac_approx,
            inv_strategy,
            inv_dtype,
            initialize_to_identity,
        )
        defaults = dict(
            lr=lr,
            damping=damping,
            T_kfac=T_kfac,
            T_inv=T_inv,
            ema_factor=ema_factor,
            kfac_approx=kfac_approx,
            inv_strategy=inv_strategy,
            inv_dtype=inv_dtype,
            initialize_to_identity=initialize_to_identity,
        )
        params = sum((list(layer.parameters()) for layer in layers), [])
        super().__init__(params, defaults)

        # initialize KFAC matrices
        self.kfacs_interior = check_layers_and_initialize_kfac(
            layers, initialize_to_identity=initialize_to_identity
        )
        self.kfacs_boundary = check_layers_and_initialize_kfac(
            layers, initialize_to_identity=initialize_to_identity
        )
        self.steps = 0
        self.inv: Dict[int, InverseKroneckerSum] = {}
        self.layers = layers

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
        loss_interior = self.evaluate_interior_loss_and_update_kfac(X_Omega, y_Omega)
        loss_interior.backward()
        loss_boundary = self.evaluate_boundary_loss_and_update_kfac(X_dOmega, y_dOmega)
        loss_boundary.backward()

        self.update_preconditioner()

        directions = []
        for layer_idx in self.kfacs_interior.keys():
            nat_grad_weight, nat_grad_bias = self.compute_natural_gradient(layer_idx)
            directions.extend([-nat_grad_weight, -nat_grad_bias])

        self._update_parameters(directions, X_Omega, y_Omega, X_dOmega, y_dOmega)

        self.steps += 1

        return loss_interior, loss_boundary

    def evaluate_interior_loss_and_update_kfac(self, X: Tensor, y: Tensor) -> Tensor:
        """Evaluate the interior loss, update the KFAC factors, and return the loss.

        Args:
            X: Interior input data.
            y: Interior label data.

        Returns:
            Differentiable interior loss.
        """
        group = self.param_groups[0]
        if self.steps % group["T_kfac"] == 0:
            loss, kfacs = evaluate_interior_loss_and_kfac_expand(self.layers, X, y)
            ema_factor = group["ema_factor"]
            for layer_idx, updates in kfacs.items():
                for destination, update in zip(self.kfacs_interior[layer_idx], updates):
                    exponential_moving_average(destination, update, ema_factor)
        else:
            loss, _, _ = evaluate_interior_loss(self.layers, X, y)

        return loss

    def evaluate_boundary_loss_and_update_kfac(self, X: Tensor, y: Tensor) -> Tensor:
        """Evaluate the boundary loss, update the KFAC factors, and return the loss.

        Args:
            X: Boundary input data.
            y: Boundary label data.

        Returns:
            Differentiable interior loss.
        """
        group = self.param_groups[0]
        if self.steps % group["T_kfac"] == 0:
            loss, kfacs = evaluate_boundary_loss_and_kfac_expand(self.layers, X, y)
            ema_factor = group["ema_factor"]
            for layer_idx, updates in kfacs.items():
                for destination, update in zip(self.kfacs_boundary[layer_idx], updates):
                    exponential_moving_average(destination, update, ema_factor)
        else:
            loss, _ = evaluate_boundary_loss(self.layers, X, y)

        return loss

    def update_preconditioner(self) -> None:
        """Update the inverse damped KFAC."""
        group = self.param_groups[0]
        T_inv = group["T_inv"]

        if self.steps % T_inv != 0:
            return

        inv_dtype = group["inv_dtype"]
        damping = group["damping"]

        # compute the KFAC inverse
        for layer_idx in self.kfacs_interior.keys():
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

            self.inv[layer_idx] = InverseKroneckerSum(
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
            [layer.weight.grad.data, layer.bias.data.unsqueeze(-1)], dim=1
        )
        nat_grad_combined = self.inv[layer_idx] @ grad_combined

        _, d_in = layer.weight.shape
        nat_grad_weight, nat_grad_bias = nat_grad_combined.split([d_in, 1], dim=1)
        return nat_grad_weight, nat_grad_bias.squeeze(1)

    @classmethod
    def _check_hyperparameters(
        cls,
        lr: Union[float, Tuple[str, List[float]]],
        damping: float,
        T_kfac: int,
        T_inv: int,
        ema_factor: float,
        kfac_approx: str,
        inv_strategy: str,
        inv_dtype: dtype,
        initialize_to_identity,
    ):
        """Check the hyperparameters for the KFAC optimizer.

        Args:
            lr: Learning rate or tuple specifying the line search.
            damping: Damping factor.
            T_kfac: Number of steps between KFAC updates.
            T_inv: Number of steps between inverse KFAC updates.
            ema_factor: Exponential moving average factor.
            kfac_approx: KFAC approximation.
            inv_strategy: Inverse strategy.
            inv_dtype: Inverse dtype.
            initialize_to_identity: Flag to initialize the inverse to the identity.

        Raises:
            ValueError: If any hyperparameter is invalid.
        """
        if T_kfac <= 0:
            raise ValueError(f"T_kfac must be positive. Got {T_kfac}.")
        if T_inv <= 0:
            raise ValueError(f"T_inv must be positive. Got {T_inv}.")
        if kfac_approx not in cls.SUPPORTED_KFAC_APPROXIMATIONS:
            raise ValueError(
                f"Unsupported KFAC approximation: {kfac_approx}. "
                + f"Supported: {cls.SUPPORTED_KFAC_APPROXIMATIONS}."
            )
        if not 0 <= ema_factor < 1:
            raise ValueError(
                "Exponential moving average factor must be in [0, 1). "
                f"Got {ema_factor}."
            )
        if isinstance(lr, float) and lr <= 0.0:
            raise ValueError(f"Learning rate must be positive. Got {lr}.")
        else:
            if lr[0] != "grid_line_search":
                raise ValueError(f"Unsupported line search: {lr[0]}.")
        if damping < 0.0:
            raise ValueError(f"Damping factor must be non-negative. Got {damping}.")
        if inv_strategy != "invert kronecker sum":
            raise ValueError(f"Unsupported inversion strategy: {inv_strategy}.")

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
            if lr[0] == "grid_line_search":

                def f() -> Tensor:
                    """Closure to evaluate the loss.

                    Returns:
                        Loss value.
                    """
                    interior_loss, _, _ = evaluate_interior_loss(
                        self.layers, X_Omega, y_Omega
                    )
                    boundary_loss, _, _ = evaluate_boundary_loss(
                        self.layers, X_dOmega, y_dOmega
                    )
                    return interior_loss + boundary_loss

                grid = lr[1]
                grid_line_search(f, params, directions, grid)

            else:
                raise ValueError(f"Unsupported line search: {lr[0]}.")
