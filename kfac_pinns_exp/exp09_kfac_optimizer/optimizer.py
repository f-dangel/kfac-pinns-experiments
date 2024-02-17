"""Implements the KFAC-for-PINNs optimizer."""

from argparse import ArgumentParser, Namespace
from typing import Dict, List, Tuple

from torch import Tensor, cat, dtype, eye, float64
from torch.nn import Module
from torch.optim import Optimizer

from kfac_pinns_exp.exp07_inverse_kronecker_sum.inverse_kronecker_sum import (
    InverseKroneckerSum,
)
from kfac_pinns_exp.exp09_kfac_optimizer.optimizer_utils import (
    check_layers_and_initialize_kfac,
    evaluate_boundary_loss,
    evaluate_boundary_loss_and_kfac_expand,
    evaluate_interior_loss,
    evaluate_interior_loss_and_kfac_expand,
)
from kfac_pinns_exp.utils import exponential_moving_average


def parse_KFACForPINNs_args(verbose: bool = False) -> Namespace:
    """Parse command-line arguments for `KFACForPINNs`.

    Args:
        verbose: Whether to print the parsed arguments. Default: `False`.

    Returns:
        A namespace with the parsed arguments.
    """
    DTYPES = {"float64": float64}
    parser = ArgumentParser(description="Parse arguments for setting up KFACForPINNs.")

    parser.add_argument(
        "--lr", type=float, help="Learning rate for the optimizer.", required=True
    )
    parser.add_argument(
        "--damping", type=float, help="Damping factor for the optimizer.", required=True
    )
    parser.add_argument(
        "--T_kfac", type=int, help="Update frequency of KFAC matrices.", default=1
    )
    parser.add_argument(
        "--T_inv",
        type=int,
        help="Update frequency of the inverse KFAC matrices.",
        default=1,
    )
    parser.add_argument(
        "--ema_factor",
        type=float,
        help="Exponential moving average factor for the KFAC matrices.",
        default=0.95,
    )
    parser.add_argument(
        "--kfac_approx",
        type=str,
        choices=KFACForPINNs.SUPPORTED_KFAC_APPROXIMATIONS,
        help="Approximation method for the KFAC matrices.",
        default="expand",
    )
    parser.add_argument(
        "--inv_strategy",
        type=str,
        choices=["invert kronecker sum"],
        help="Inversion strategy for KFAC.",
        default="invert kronecker sum",
    )
    parser.add_argument(
        "--inv_dtype",
        type=str,
        choices=DTYPES.keys(),
        help="Data type for the inverse KFAC matrices.",
        default="float64",
    )

    args, _ = parser.parse_known_args()
    # overwrite inv_dtype with value from dictionary
    args.inv_dtype = DTYPES[args.inv_dtype]

    if verbose:
        print("Parsed arguments for KFACForPINNs: ", args)

    return args


class KFACForPINNs(Optimizer):
    """KFAC optimizer for PINN problems."""

    SUPPORTED_KFAC_APPROXIMATIONS = {"expand"}

    def __init__(
        self,
        layers: List[Module],
        lr: float,
        damping: float,
        T_kfac: int = 1,
        T_inv: int = 1,
        ema_factor: float = 0.95,
        kfac_approx: str = "expand",
        inv_strategy: str = "invert kronecker sum",
        inv_dtype: dtype = float64,
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
            lr: Learning rate. Must be positive.
            damping: Damping factor. Must be positive.
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
                data type as the parameters after the inversio.

        Raises:
            ValueError: If any of the hyper-parameters is invalid.
        """
        # check hyper-parameters
        if kfac_approx not in self.SUPPORTED_KFAC_APPROXIMATIONS:
            raise ValueError(
                f"Unsupported KFAC approximation: {kfac_approx}. "
                + f"Supported: {self.SUPPORTED_KFAC_APPROXIMATIONS}."
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
        if inv_strategy != "invert kronecker sum":
            raise ValueError(
                f"Unsupported inversion strategy: {inv_strategy}. "
                + "Supported: 'invert kronecker sum'."
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
        )
        params = sum((list(layer.parameters()) for layer in layers), [])
        super().__init__(params, defaults)

        # initialize KFAC matrices
        self.kfacs_interior = check_layers_and_initialize_kfac(
            layers, initialize_to_identity=True
        )
        self.kfacs_boundary = check_layers_and_initialize_kfac(
            layers, initialize_to_identity=True
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

        group = self.param_groups[0]
        lr = group["lr"]

        for layer_idx in self.kfacs_interior.keys():
            nat_grad_weight, nat_grad_bias = self.compute_natural_gradient(layer_idx)
            layer = self.layers[layer_idx]
            layer.weight.data.sub_(nat_grad_weight, alpha=lr)
            layer.bias.data.sub_(nat_grad_bias, alpha=lr)

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
            loss, _ = evaluate_interior_loss(self.layers, X, y)

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
