"""Implements the KFAC-for-PINNs optimizer."""

from argparse import ArgumentParser, Namespace
from math import sqrt
from typing import Dict, List, Tuple, Union

from backpack.hessianfree.rop import jacobian_vector_product
from torch import Tensor, arange, cat, dtype, float64, tensor
from torch.nn import Module
from torch.optim import Optimizer

from kfac_pinns_exp import (
    fokker_planck_isotropic_equation,
    heat_equation,
    log_fokker_planck_isotropic_equation,
    poisson_equation,
)
from kfac_pinns_exp.inverse_kronecker_sum import InverseKroneckerSum
from kfac_pinns_exp.kfac_utils import check_layers_and_initialize_kfac
from kfac_pinns_exp.optim.engd import ENGD_DEFAULT_LR
from kfac_pinns_exp.optim.line_search import (
    grid_line_search,
    parse_grid_line_search_args,
)
from kfac_pinns_exp.parse_utils import parse_known_args_and_remove_from_argv
from kfac_pinns_exp.pinn_utils import (
    evaluate_boundary_loss,
    evaluate_boundary_loss_and_kfac,
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
        f"--{prefix}damping_heuristic",
        type=str,
        choices=KFAC.SUPPORTED_DAMPING_HEURISTICS,
        help="How to distribute the damping onto the two Kronecker factors.",
        default="same",
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
    if getattr(args, lr) == "auto":
        # use a small learning rate for the first step
        lr_init = 1e-6
        setattr(args, lr, (getattr(args, lr), lr_init))

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
            Poisson (`'poisson'`), heat (`'heat'`) and Fokker-Planck equation with
            isotropic diffusivity and vector field (`'fokker-planck-isotropic'`).
        SUPPORTED_DAMPING_HEURISTICS: Available damping heuristics how to distribute
            the damping onto the two Kronecker factors. Currently supports `'same'`
            and `'trace-norm'` (from Martens et al. (2015), Section 6.3 in
            https://arxiv.org/pdf/1503.05671).
        LOSS_AND_KFAC_EVALUATORS: Dictionary from equation and loss type to functions
            that evaluate the loss and KFAC matrices on a batch.
        LOSS_EVALUATORS: Dictionary from equation and loss type to functions that
            evaluate the loss on a batch.
    """

    LOSS_AND_KFAC_EVALUATORS = {
        "poisson": {
            "interior": poisson_equation.evaluate_interior_loss_and_kfac,
            "boundary": evaluate_boundary_loss_and_kfac,
        },
        "heat": {
            "interior": heat_equation.evaluate_interior_loss_and_kfac,
            "boundary": evaluate_boundary_loss_and_kfac,
        },
        "fokker-planck-isotropic": {
            "interior": fokker_planck_isotropic_equation.evaluate_interior_loss_and_kfac,  # noqa: B950
            "boundary": evaluate_boundary_loss_and_kfac,
        },
        "log-fokker-planck-isotropic": {
            "interior": log_fokker_planck_isotropic_equation.evaluate_interior_loss_and_kfac,  # noqa: B950
            "boundary": evaluate_boundary_loss_and_kfac,
        },
    }
    LOSS_EVALUATORS = {
        "poisson": {
            "interior": poisson_equation.evaluate_interior_loss,
            "boundary": evaluate_boundary_loss,
        },
        "heat": {
            "interior": heat_equation.evaluate_interior_loss,
            "boundary": evaluate_boundary_loss,
        },
        "fokker-planck-isotropic": {
            "interior": fokker_planck_isotropic_equation.evaluate_interior_loss,
            "boundary": evaluate_boundary_loss,
        },
        "log-fokker-planck-isotropic": {
            "interior": log_fokker_planck_isotropic_equation.evaluate_interior_loss,
            "boundary": evaluate_boundary_loss,
        },
    }
    SUPPORTED_KFAC_APPROXIMATIONS = {"expand", "reduce"}
    SUPPORTED_GGN_TYPES = {"type-2", "empirical", "forward-only"}
    SUPPORTED_EQUATIONS = {
        "poisson",
        "heat",
        "fokker-planck-isotropic",
        "log-fokker-planck-isotropic",
    }
    SUPPORTED_DAMPING_HEURISTICS = {"same", "trace-norm"}

    def __init__(
        self,
        layers: List[Module],
        damping: float,
        lr: Union[float, Tuple[str, Union[List[float], float]]] = ENGD_DEFAULT_LR,
        T_kfac: int = 1,
        T_inv: int = 1,
        ema_factor: float = 0.95,
        kfac_approx: str = "expand",
        inv_strategy: str = "invert kronecker sum",
        ggn_type: str = "type-2",
        inv_dtype: dtype = float64,
        initialize_to_identity: bool = False,
        equation: str = "poisson",
        damping_heuristic: str = "same",
        momentum: float = 0.0,
    ) -> None:
        """Set up the optimizer.

        Limitations:
            - No parameter group support. Can only train all parameters.

        Args:
            layers: List of layers of the neural network.
            damping: Damping factor. Must be positive.
            lr: The learning rate, line search or momentum strategy:
                - If a float, this will be used as constant learning rate.
                - If a tuple of the form `('grid_line_search', grid)` with grid a list
                  of step size candidates, the optimizer will perform a grid search
                  along the update direction and choose the best candidate.
                - If a tuple of the form `('auto', init_lr)`, the optimizer will auto-
                  matically determine the learning rate and momentum at each iteration
                  and use the initial learning rate for the first step, as the
                  heuristic depends on the previous update. See Section 7 of the KFAC
                  paper (https://arxiv.org/pdf/1503.05671) for details.
            T_kfac: Positive integer specifying the update frequency for
                the boundary and the interior terms' KFACs. Default is `1`.
            T_inv: Positive integer specifying the preconditioner update
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
                `torch.float64`. The preconditioner will be converted back to the same
                data type as the parameters after the inversion.
            initialize_to_identity: Whether to initialize the KFAC factors to the
                identity matrix. Default is `False` (initialize with zero).
            equation: Equation to solve. Currently supports `'poisson'`, `'heat'`, and
                `'fokker-planck-isotropic'`. Default: `'poisson'`.
            damping_heuristic: How to distribute the damping onto the two Kronecker
                factors. Currently supports `'same'` and `'trace-norm` (see Section 6.3
                of https://arxiv.org/pdf/1503.05671). Default is `'same'`.
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
            damping_heuristic=damping_heuristic,
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
        damping_heuristic = group["damping_heuristic"]

        # compute the KFAC inverse
        for layer_idx in self.layer_idxs:
            # NOTE that in the literature (column-stacking), KFAC w.r.t. the flattened
            # weights is A₁ ⊗ A₂ + B₁ ⊗ B₂. However, in code we use row-stacking
            # flattening. Effectively, we have to swap the Kronecker factors to obtain
            # KFAC w.r.t. the flattened (row-stacking) weights.
            A2, A1 = self.kfacs_interior[layer_idx]
            B2, B1 = self.kfacs_boundary[layer_idx]

            A2, A1 = self.add_damping(A2, A1, damping, damping_heuristic)
            B2, B1 = self.add_damping(B2, B1, damping, damping_heuristic)

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
        elif lr[0] in {"grid_line_search", "auto"}:
            if lr[0] == "auto":
                lr_init = lr[1]
                if lr_init <= 0:
                    raise ValueError(
                        f"Initial learning rate must be positive. Got {lr_init}."
                    )
                momentum = group["momentum"]
                if momentum != 0.0:
                    raise ValueError(
                        f"Momentum was specified to non-zero value {momentum}"
                        + "although automatic learning rate and momentum is enabled."
                    )
        else:
            raise ValueError(f"Unsupported line search: {lr[0]}.")

        damping = group["damping"]
        if damping < 0.0:
            raise ValueError(f"Damping factor must be non-negative. Got {damping}.")

        inv_strategy = group["inv_strategy"]
        if inv_strategy != "invert kronecker sum":
            raise ValueError(f"Unsupported inversion strategy: {inv_strategy}.")

        damping_heuristic = group["damping_heuristic"]
        if damping_heuristic not in self.SUPPORTED_DAMPING_HEURISTICS:
            raise ValueError(
                f"Unsupported damping heuristic: {damping_heuristic}. "
                + f"Supported: {self.SUPPORTED_DAMPING_HEURISTICS}."
            )

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
        group = self.param_groups[0]
        lr = group["lr"]
        params = group["params"]

        if isinstance(lr, float):
            self.add_momentum(directions)
            for param, direction in zip(params, directions):
                param.data.add_(direction, alpha=lr)
        else:
            if lr[0] == "grid_line_search":
                self.add_momentum(directions)

                def f() -> Tensor:
                    """Closure to evaluate the loss.

                    Returns:
                        Loss value.
                    """
                    interior_loss = self.eval_loss(X_Omega, y_Omega, "interior")
                    boundary_loss = self.eval_loss(X_dOmega, y_dOmega, "boundary")
                    return interior_loss + boundary_loss

                grid = lr[1]
                grid_line_search(f, params, directions, grid)

            elif lr[0] == "auto":  # KFAC heuristic for auto learning rate & momentum
                if self.steps == 0:  # use the second value as initial learning rate
                    alpha = lr[1]
                    updates = [d.mul_(alpha) for d in directions]
                else:  # use heuristic
                    previous = [self.state[p]["previous_update"] for p in params]
                    alpha, mu = self.auto_lr_and_momentum(
                        directions, previous, X_Omega, y_Omega, X_dOmega, y_dOmega
                    )
                    updates = [
                        d.mul_(alpha).add_(prev.mul_(mu))
                        for d, prev in zip(directions, previous)
                    ]
                for p, u in zip(params, updates):
                    self.state[p]["previous_update"] = u
                    p.data.add_(u)

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
        loss_evaluator = self.LOSS_EVALUATORS[self.equation][loss_type]
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
        loss_and_kfac_evaluator = self.LOSS_AND_KFAC_EVALUATORS[self.equation][
            loss_type
        ]
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

    @staticmethod
    def add_damping(
        A: Tensor, B: Tensor, damping: float, heuristic: str
    ) -> Tuple[Tensor, Tensor]:
        """Add damping to the KFAC factors.

        Args:
            A: The input-based Kronecker factor.
            B: The output-gradient-based Kronecker factor.
            damping: The damping factor.
            heuristic: The damping heuristic.

        Returns:
            A tuple of the damped KFAC factors.

        Raises:
            ValueError: If the damping heuristic is not supported.
        """
        (dim_A,), (dim_B,) = set(A.shape), set(B.shape)

        if heuristic == "same":
            damping_A, damping_B = damping, damping
        elif heuristic == "trace-norm":
            # trace-norm heuristic from Martens et al. (2015),
            # see https://arxiv.org/pdf/1503.05671, Sectin 6.3
            pi = ((A.trace() * dim_B) / (B.trace() * dim_A)).sqrt()
            damping_A = sqrt(damping) * pi
            damping_B = sqrt(damping) / pi
        else:
            raise ValueError(f"Unsupported damping heuristic: {heuristic}.")

        A_damped = A.clone()
        idx_A = arange(dim_A, device=A.device)
        A_damped[idx_A, idx_A] = A_damped.diag().add_(damping_A)

        B_damped = B.clone()
        idx_B = arange(dim_B, device=B.device)
        B_damped[idx_B, idx_B] = B_damped.diag().add_(damping_B)

        return A_damped, B_damped

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

    def auto_lr_and_momentum(
        self,
        direction: List[Tensor],
        previous: List[Tensor],
        X_Omega: Tensor,
        y_Omega: Tensor,
        X_dOmega: Tensor,
        y_dOmega: Tensor,
    ) -> Tuple[float, float]:
        """Automatically choose learning rate and momentum for the current step.

        See KFAC paper (https://arxiv.org/pdf/1503.05671), Section 7. Minimizes the
        two-dimensional quadratic model with the true Gramian along the current update
        direction and the previous update step.

        Args:
            direction: The update direction from multiplying the inverse KFAC onto the
                negative gradient.
            previous: The parameter update from the previous iteration.
            X_Omega: Input data to the interior loss.
            y_Omega: Target data to the interior loss.
            X_dOmega: Input data to the boundary loss.
            y_dOmega: Target data to the boundary loss.

        Returns:
            The learning rate and momentum for the current step.
        """
        group = self.param_groups[0]
        damping = group["damping"]  # = λ + η in the KFAC paper
        params = sum((list(layer.parameters()) for layer in self.layers), [])
        d, D = previous, direction  # = δ, Δ in the KFAC paper
        g = [p.grad for p in params]  # = ∇h in the KFAC paper

        DD = sum((D_**2).sum() for D_ in D)  # = ||Δ||₂²
        dd = sum((d_**2).sum() for d_ in d)  # = ||δ||₂²
        dD = sum((d_ * D_).sum() for d_, D_ in zip(d, D))  # = δᵀΔ
        gd = sum((g_ * d_).sum() for g_, d_ in zip(g, d))  # = ∇hᵀδ
        gD = sum((g_ * D_).sum() for g_, D_ in zip(g, D))  # = ∇hᵀΔ

        # compute Gramian-vector products along δ and Δ
        eval_interior_loss = self.LOSS_EVALUATORS[self.equation]["interior"]
        eval_boundary_loss = self.LOSS_EVALUATORS[self.equation]["boundary"]

        # multiply with the boundary Gramian and add
        _, residual, _ = eval_interior_loss(self.layers, X_Omega, y_Omega)
        # correct for batch size reduction factor
        residual /= sqrt(residual.shape[0])
        (Jd,) = jacobian_vector_product(residual, params, d, retain_graph=True)
        (JD,) = jacobian_vector_product(residual, params, D, retain_graph=False)
        DGD = (JD**2).sum()  # = Δᵀ G_Ω Δ
        dGd = (Jd**2).sum()  # = δᵀ G_Ω δ
        dGD = (Jd * JD).sum()  # = δᵀ G_Ω Δ

        _, residual, _ = eval_boundary_loss(self.layers, X_dOmega, y_dOmega)
        # correct for batch size reduction factor
        residual /= sqrt(residual.shape[0])
        (Jd,) = jacobian_vector_product(residual, params, d, retain_graph=True)
        (JD,) = jacobian_vector_product(residual, params, D, retain_graph=False)
        DGD += (JD**2).sum()  # Δᵀ (G_Ω + G_∂Ω) Δ
        dGd += (Jd**2).sum()  # δᵀ (G_Ω + G_∂Ω) δ
        dGD += (Jd * JD).sum()  # δᵀ (G_Ω + G_∂Ω) Δ

        # solve the 2x2 linear system from page 28 (https://arxiv.org/pdf/1503.05671)
        # for the learning rate and momentum
        A = tensor(
            [
                [DGD + damping * DD, dGD + damping * dD],
                [dGD + damping * dD, dGd + damping * dd],
            ]
        )
        b = tensor([gD, gd])
        (lr, momentum) = -(A.inverse() @ b)

        return lr.item(), momentum.item()
