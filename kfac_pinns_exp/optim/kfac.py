"""Implements the KFAC-for-PINNs optimizer."""

from argparse import ArgumentParser, Namespace
from typing import Dict, List, Tuple, Union

from torch import Tensor, cat, dtype, eye, float64, kron, zeros
from torch.nn import Module, Sequential
from torch.optim import Optimizer

from kfac_pinns_exp.exp07_inverse_kronecker_sum.inverse_kronecker_sum import (
    InverseKroneckerSum,
)
from kfac_pinns_exp.kfac_utils import (
    check_layers_and_initialize_kfac,
    gramian_basis_to_kfac_basis,
)
from kfac_pinns_exp.optim.engd import ENGD_DEFAULT_LR
from kfac_pinns_exp.optim.line_search import (
    grid_line_search,
    parse_grid_line_search_args,
)
from kfac_pinns_exp.parse_utils import parse_known_args_and_remove_from_argv
from kfac_pinns_exp.poisson_equation import (
    evaluate_boundary_loss,
    evaluate_boundary_loss_and_kfac_expand,
    evaluate_interior_gramian,
    evaluate_interior_loss,
    evaluate_interior_loss_and_kfac_expand,
)
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
        f"--{prefix}USE_EXACT_BOUNDARY_GRAMIAN",
        action="store_true",
        default=False,
        help="Whether to use the exact boundary Gramian.",
    )
    parser.add_argument(
        f"--{prefix}USE_EXACT_INTERIOR_GRAMIAN",
        action="store_true",
        default=False,
        help="Whether to use the exact interior Gramian.",
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
        # debugging flags
        USE_EXACT_BOUNDARY_GRAMIAN: bool = False,
        USE_EXACT_INTERIOR_GRAMIAN: bool = False,
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

        # set debuggin flags
        self.USE_EXACT_BOUNDARY_GRAMIAN = USE_EXACT_BOUNDARY_GRAMIAN
        self.USE_EXACT_INTERIOR_GRAMIAN = USE_EXACT_INTERIOR_GRAMIAN

        # initialize KFAC matrices or Gramians for the interior term
        if self.USE_EXACT_INTERIOR_GRAMIAN:
            (dev,) = {p.device for p in params}
            (dt,) = {p.dtype for p in params}
            kwargs = {"device": dev, "dtype": dt}
            block_sizes = {
                idx: sum(p.numel() for p in layer.parameters())
                for idx, layer in enumerate(layers)
                if list(layer.parameters())
            }
            self.gramians_interior = {
                idx: (
                    eye(size, **kwargs)
                    if initialize_to_identity
                    else zeros(size, size, **kwargs)
                )
                for idx, size in block_sizes.items()
            }
        else:
            self.kfacs_interior = check_layers_and_initialize_kfac(
                layers, initialize_to_identity=initialize_to_identity
            )

        # initialize KFAC matrices or Gramians for the boundary term
        if self.USE_EXACT_BOUNDARY_GRAMIAN:
            raise NotImplementedError
        else:
            self.kfacs_boundary = check_layers_and_initialize_kfac(
                layers, initialize_to_identity=initialize_to_identity
            )

        self.steps = 0
        self.inv: Dict[int, Union[InverseKroneckerSum, Tensor]] = {}
        self.layers = layers

        if self.USE_EXACT_BOUNDARY_GRAMIAN or self.USE_EXACT_INTERIOR_GRAMIAN:
            self.model = Sequential(*layers)

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
        if not self.USE_EXACT_BOUNDARY_GRAMIAN:
            layer_idxs = self.kfacs_boundary.keys()
        elif not self.USE_EXACT_INTERIOR_GRAMIAN:
            layer_idxs = self.kfacs_interior.keys()
        else:
            raise NotImplementedError

        for layer_idx in layer_idxs:
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
            ema_factor = group["ema_factor"]
            if self.USE_EXACT_INTERIOR_GRAMIAN:
                gramian = evaluate_interior_gramian(self.model, X, "per_layer")
                loss, _, _ = evaluate_interior_loss(self.model, X, y)
                for destination, update in zip(
                    self.gramians_interior.values(), gramian
                ):
                    exponential_moving_average(destination, update, ema_factor)
            else:
                loss, kfacs = evaluate_interior_loss_and_kfac_expand(self.layers, X, y)
                for layer_idx, updates in kfacs.items():
                    for destination, update in zip(
                        self.kfacs_interior[layer_idx], updates
                    ):
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
            if self.USE_EXACT_BOUNDARY_GRAMIAN:
                raise NotImplementedError
            loss, kfacs = evaluate_boundary_loss_and_kfac_expand(self.layers, X, y)
            ema_factor = group["ema_factor"]
            for layer_idx, updates in kfacs.items():
                for destination, update in zip(self.kfacs_boundary[layer_idx], updates):
                    exponential_moving_average(destination, update, ema_factor)
        else:
            loss, _, _ = evaluate_boundary_loss(self.layers, X, y)

        return loss

    def update_preconditioner(self) -> None:
        """Update the inverse damped KFAC."""
        group = self.param_groups[0]
        T_inv = group["T_inv"]

        if self.steps % T_inv != 0:
            return

        inv_dtype = group["inv_dtype"]
        damping = group["damping"]

        # compute matrix representation and invert
        if self.USE_EXACT_BOUNDARY_GRAMIAN or self.USE_EXACT_INTERIOR_GRAMIAN:
            if self.USE_EXACT_BOUNDARY_GRAMIAN:
                raise NotImplementedError
            else:
                gramians_boundary = {
                    idx: kron(B, A) for idx, (A, B) in self.kfacs_boundary.items()
                }
            if self.USE_EXACT_INTERIOR_GRAMIAN:
                # The basis of KFAC is `(W, b).flatten()` but the Gramian's basis
                # is `(flatten(W).T, b.T).T`. We need to re-arrange the Gramian to
                # match the basis of KFAC.
                dims = [
                    (self.layers[idx].weight.shape[1], self.layers[idx].weight.shape[0])
                    for idx in self.gramians_interior
                ]
                assert len(dims) == len(self.gramians_interior)
                gramians_interior = {
                    idx: gramian_basis_to_kfac_basis(g, dim_A, dim_B)
                    for (idx, g), (dim_A, dim_B) in zip(
                        self.gramians_interior.items(), dims
                    )
                }
            else:
                raise NotImplementedError

            gramians = {
                idx: gramians_interior[idx] + gramians_boundary[idx]
                for idx in gramians_interior
            }
            # add damping
            gramians = {
                idx: g + damping * eye(g.shape[0], dtype=g.dtupe, device=g.device)
                for idx, g in gramians.items()
            }
            for idx, g in gramians.items():
                self.inv[idx] = g.inverse()
            return

        # compute the KFAC inverse
        for layer_idx in self.kfacs_interior.keys():
            weight_dtype = self.layers[layer_idx].weight.dtype
            weight_device = self.layers[layer_idx].weight.device

            if (
                not self.USE_EXACT_BOUNDARY_GRAMIAN
                and not self.USE_EXACT_INTERIOR_GRAMIAN
            ):
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
            else:
                raise NotImplementedError

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
        d_out, d_in = layer.weight.shape
        if self.USE_EXACT_BOUNDARY_GRAMIAN or self.USE_EXACT_INTERIOR_GRAMIAN:
            nat_grad_combined = (self.inv[layer_idx] @ grad_combined.flatten()).reshape(
                d_out, d_in + 1
            )
        else:
            nat_grad_combined = self.inv[layer_idx] @ grad_combined

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
        if isinstance(lr, float):
            if lr <= 0.0:
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
