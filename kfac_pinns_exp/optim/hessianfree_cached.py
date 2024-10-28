"""Argument parser and implementation of Hessian-free optimizer+cached linear operators.

The implementation is at https://github.com/ltatzel/PyTorchHessianFree.
"""

from argparse import ArgumentParser, Namespace
from typing import Callable, Dict, List, Tuple, Union

from hessianfree.optimizer import HessianFree
from torch import Tensor, cat, dtype, float64, no_grad
from torch.nn import Linear, Module

from kfac_pinns_exp.inverse_kronecker_sum import InverseKroneckerSum
from kfac_pinns_exp.kfac_utils import (
    check_layers_and_initialize_kfac,
    compute_kronecker_factors,
)
from kfac_pinns_exp.linops import GramianLinearOperator
from kfac_pinns_exp.optim.kfac import KFAC
from kfac_pinns_exp.parse_utils import parse_known_args_and_remove_from_argv
from kfac_pinns_exp.utils import exponential_moving_average


def parse_HessianFreeCached_args(
    verbose: bool = True, prefix: str = "HessianFreeCached_"
) -> Namespace:
    """Parse command-line arguments for the cached Hessian-free optimizer.

    Args:
        verbose: Whether to print the parsed arguments. Default: `True`.
        prefix: Prefix for the arguments. Default: `"HessianFreeCached_"`.

    Returns:
        A namespace with the parsed arguments.
    """
    DTYPES = {"float64": float64}
    parser = ArgumentParser(description="Cached Hessian-free optimizer parameters.")
    parser.add_argument(
        f"--{prefix}equation",
        type=str,
        choices=HessianFreeCached.SUPPORTED_EQUATIONS,
        help="The equation to solve.",
        default="poisson",
    )
    parser.add_argument(
        f"--{prefix}damping",
        type=float,
        default=1.0,
        help="Tikhonov damping parameter.",
    )
    parser.add_argument(
        f"--{prefix}no_adapt_damping",
        dest=f"{prefix}adapt_damping",
        default=True,
        action="store_false",
        help="Whether to deactivate adaptive damping and use constant damping.",
    )
    parser.add_argument(
        f"--{prefix}cg_max_iter",
        type=int,
        default=250,
        help="Maximum number of CG iterations.",
    )
    parser.add_argument(
        f"--{prefix}cg_decay_x0",
        type=float,
        default=0.95,
        help="Decay factor of the previous CG solution used as init for the next.",
    )
    parser.add_argument(
        f"--{prefix}no_use_cg_backtracking",
        dest=f"{prefix}use_cg_backtracking",
        default=True,
        action="store_false",
        help="Whether to disable CG backtracking.",
    )
    parser.add_argument(
        f"--{prefix}lr",
        type=float,
        default=1.0,
        help="Learning rate.",
    )
    parser.add_argument(
        f"--{prefix}no_use_linesearch",
        dest=f"{prefix}use_linesearch",
        default=True,
        action="store_false",
        help="Whether to disable line search",
    )
    parser.add_argument(
        f"--{prefix}verbose",
        dest=f"{prefix}verbose",
        default=False,
        action="store_true",
        help="Whether to print internals to the command line.",
    )
    parser.add_argument(
        f"--{prefix}approximation",
        choices=HessianFreeCached.SUPPORTED_APPROXIMATIONS,
        default="full",
        help="The Gramian approximation to use.",
    )
    parser.add_argument(
        f"--{prefix}kfac_preconditioner",
        default=False,
        action="store_true",
        help="Whether to precondition the optimizer with KFAC.",
    )
    parser.add_argument(
        f"--{prefix}kfac_preconditioner_initialize_to_identity",
        default=False,
        action="store_true",
        help="Whether to initialize the KFAC preconditioner to the identity.",
    )
    parser.add_argument(
        f"--{prefix}kfac_preconditioner_ema_factor",
        type=float,
        default=0.95,
        help="Exponential moving average factor for the KFAC preconditioner.",
    )
    parser.add_argument(
        f"--{prefix}kfac_preconditioner_T_kfac",
        type=int,
        default=1,
        help="Update period for the pre-conditioner's KFAC matrices.",
    )
    parser.add_argument(
        f"--{prefix}kfac_preconditioner_T_inv",
        type=int,
        default=1,
        help="Update period for the pre-conditioner's inverse KFAC matrices.",
    )
    parser.add_argument(
        f"--{prefix}kfac_preconditioner_kfac_approx",
        type=str,
        default="expand",
        choices=KFAC.SUPPORTED_KFAC_APPROXIMATIONS,
        help="The KFAC approximation to use.",
    )

    parser.add_argument(
        f"--{prefix}kfac_preconditioner_inv_dtype",
        type=str,
        choices=DTYPES.keys(),
        help="Data type for the inverse KFAC matrices.",
        default="float64",
    )

    args = parse_known_args_and_remove_from_argv(parser)
    # overwrite inv_dtype with value from dictionary
    inv_dtype = f"{prefix}kfac_preconditioner_inv_dtype"
    setattr(args, inv_dtype, DTYPES[getattr(args, inv_dtype)])

    if verbose:
        print(f"Hessian-free arguments: {args}")

    return args


class HessianFreeCached(HessianFree):
    """Hessian-free optimizer with cached linear operators.

    Uses linear operators which pre-compute the information required to multiply
    with the curvature matrix. This is different to `HessianFree`, which uses
    nested automatic differentiation (should be slower).

    Attributes:
        SUPPORTED_APPROXIMATIONS: The supported Gramian approximations.
    """

    SUPPORTED_APPROXIMATIONS = {"full", "per_layer"}
    SUPPORTED_EQUATIONS = {
        "poisson",
        "heat",
        "fokker-planck-isotropic",
        "log-fokker-planck-isotropic",
    }

    def __init__(
        self,
        layers: List[Module],
        equation: str,
        damping: float = 1.0,
        adapt_damping: bool = True,
        cg_max_iter: int = 250,
        cg_decay_x0: float = 0.95,
        use_cg_backtracking: bool = True,
        lr: float = 1.0,
        use_linesearch: bool = True,
        verbose: bool = False,
        approximation: str = "full",
        kfac_preconditioner: bool = False,
        kfac_preconditioner_initialize_to_identity: bool = False,
        kfac_preconditioner_ema_factor: float = 0.95,
        kfac_preconditioner_T_kfac: int = 1,
        kfac_preconditioner_T_inv: int = 1,
        kfac_preconditioner_kfac_approx: str = "expand",
        kfac_preconditioner_inv_dtype: dtype = float64,
    ):
        """Initialize the Hessian-free optimizer with cached linear operators.

        Args:
            layers: A list of PyTorch modules representing the neural network.
            equation: A string specifying the PDE.
            damping: Initial Tikhonov damping parameter. Default: `1.0`.
            adapt_damping: Whether to adapt the damping parameter. Default: `True`.
            cg_max_iter: Maximum number of CG iterations. Default: `250`.
            cg_decay_x0: Decay factor of the previous CG solution used as next init.
                Default: `0.95`.
            use_cg_backtracking: Whether to use CG backtracking. Default: `True`.
            lr: Learning rate. Default: `1.0`.
            use_linesearch: Whether to use line search. Default: `True`.
            verbose: Whether to print internals to the command line. Default: `False`.
            approximation: The Gramian approximation to use. Can be `'full'` or
                `'per_layer'`. Default: `'full'`.
            kfac_preconditioner: Whether to precondition the linear system solved by
                CG with KFAC. Default: `False`.
            kfac_preconditioner_initialize_to_identity: Only relevant if
                `kfac_preconditioner=True`. Whether to initialize the KFAC
                pre-conditioner to the identity matrix. Default: `False`.
            kfac_preconditioner_ema_factor: Only relevant if `kfac_preconditioner=True`.
                The exponential moving average factor for the KFAC pre-conditioner.
                Default: `0.95`.
            kfac_preconditioner_T_kfac: Only relevant if `kfac_preconditioner=True`.
                The update frequency of the pre-conditioner's KFAC matrices.
                Default: `1`.
            kfac_preconditioner_T_inv: Only relevant if `kfac_preconditioner=True`.
                The update frequency of the pre-conditioner's inverse KFAC matrices.
                Default: `1`.
            kfac_preconditioner_kfac_approx: Only relevant if
                `kfac_preconditioner=True`. The approximation to use for the KFAC
                matrices. Can be `'expand'` or `'reduce'`. Default: `'expand'`.
            kfac_preconditioner_inv_dtype: Only relevant if
                `kfac_preconditioner=True`. The precision in which to carry out the
                inversion of the pre-conditioner's KFAC matrices. Default: `float64`.

        Raises:
            NotImplementedError: If the trainable parameters are not in linear layers.
            ValueError: If approximation or equation are not supported.
        """
        if approximation not in self.SUPPORTED_APPROXIMATIONS:
            raise ValueError(
                f"Approximation {approximation!r} not supported."
                f"Supported approximations: {self.SUPPORTED_APPROXIMATIONS}."
            )
        if equation not in self.SUPPORTED_EQUATIONS:
            raise ValueError(
                f"Equation {equation!r} not supported."
                f"Supported equations: {self.SUPPORTED_EQUATIONS}."
            )

        self.layer_idxs = []
        for idx, layer in enumerate(layers):
            if isinstance(layer, Linear):
                if (
                    layer.weight.requires_grad
                    and layer.bias is not None
                    and layer.bias.requires_grad
                ):
                    self.layer_idxs.append(idx)
                elif any(p.requires_grad for p in layer.parameters()):
                    raise NotImplementedError(
                        "Trainable linear layers must have differentiable weight+bias."
                    )
            elif any(p.requires_grad for p in layer.parameters()):
                raise NotImplementedError(
                    "Trainable parameters must be in linear layers."
                )
        self.params = sum(
            (list(layers[idx].parameters()) for idx in self.layer_idxs), []
        )

        super().__init__(
            self.params,
            curvature_opt="ggn",
            damping=damping,
            adapt_damping=adapt_damping,
            cg_max_iter=cg_max_iter,
            cg_decay_x0=cg_decay_x0,
            use_cg_backtracking=use_cg_backtracking,
            lr=lr,
            use_linesearch=use_linesearch,
            verbose=verbose,
        )

        self.global_step = 0
        self.layers = layers
        self.equation = equation
        self.approximation = approximation
        self.kfac_preconditioner = kfac_preconditioner

        # initialize Kronecker factors
        if self.kfac_preconditioner:
            self.kfacs_interior = check_layers_and_initialize_kfac(
                layers,
                initialize_to_identity=kfac_preconditioner_initialize_to_identity,
            )
            self.kfacs_boundary = check_layers_and_initialize_kfac(
                layers,
                initialize_to_identity=kfac_preconditioner_initialize_to_identity,
            )
            self.kfac_preconditioner_inv: Dict[
                int, Union[InverseKroneckerSum, Tensor]
            ] = {}
            self.kfac_preconditioner_ema_factor = kfac_preconditioner_ema_factor
            self.kfac_preconditioner_T_kfac = kfac_preconditioner_T_kfac
            self.kfac_preconditioner_T_inv = kfac_preconditioner_T_inv
            self.kfac_preconditioner_kfac_approx = kfac_preconditioner_kfac_approx
            self.kfac_preconditioner_inv_dtype = kfac_preconditioner_inv_dtype

    def step(
        self,
        # linear operator specific arguments
        X_Omega: Tensor,
        y_Omega: Tensor,
        X_dOmega: Tensor,
        y_dOmega: Tensor,
        # remaining arguments from parent class
        forward: Callable[[Tensor], Tuple[Tensor, Tensor]],
        test_deterministic: bool = False,
    ) -> Tensor:
        """Perform a single optimization step.

        Args:
            X_Omega: The input data for the interior loss.
            y_Omega: The target data for the interior loss.
            X_dOmega: The input data for the boundary loss.
            y_dOmega: The target data for the boundary loss.
            forward: A function that computes the loss and the model's output.
            test_deterministic: Whether to test the deterministic behavior of `forward`.
                Default is `False`.

        Returns:
            The loss after the optimization step.
        """
        linop_interior = GramianLinearOperator(
            self.equation, self.layers, X_Omega, y_Omega, "interior"
        )
        linop_boundary = GramianLinearOperator(
            self.equation, self.layers, X_dOmega, y_dOmega, "boundary"
        )
        grad = cat(
            [
                (g_int.detach() + g_bnd.detach()).flatten()
                for g_int, g_bnd in zip(linop_interior.grad, linop_boundary.grad)
            ]
        )
        del linop_interior.grad, linop_boundary.grad  # remove to save memory

        @no_grad()
        def mvp(v: Tensor) -> Tensor:
            """Multiply the Gramian onto a vector.

            Args:
                v: A vector to multiply the Gramian with.

            Returns:
                The product of the Gramian with the vector.
            """
            return linop_interior @ v + linop_boundary @ v

        # update pre-conditioner
        if self.kfac_preconditioner:
            if self.global_step % self.kfac_preconditioner_T_kfac == 0:
                self._update_kfac(linop_interior)
                self._update_kfac(linop_boundary)
            if self.global_step % self.kfac_preconditioner_T_inv == 0:
                self._update_inv_kfac()

        result = super().step(
            forward,
            grad=grad,
            mvp=mvp,
            M_func=self._apply_inv_kfac if self.kfac_preconditioner else None,
            test_deterministic=test_deterministic,
        )
        self.global_step += 1
        return result

    def _update_kfac(self, linop: GramianLinearOperator):
        """Update the Kronecker factors for a Gramian.

        In-place updates either `self.kfacs_interior` or `self.kfacs_boundary` depending
        on the linear operator's `loss_type` attribute.

        Args:
            linop: The linear operator of an interior or boundary Gramian.
        """
        ema_factor = self.kfac_preconditioner_ema_factor

        target = {"interior": self.kfacs_interior, "boundary": self.kfacs_boundary}[
            linop.loss_type
        ]
        kfacs = compute_kronecker_factors(
            self.layers,
            linop.layer_inputs,
            linop.layer_grad_outputs,
            ggn_type="type-2",
            kfac_approx=self.kfac_preconditioner_kfac_approx,
        )

        for layer_idx, (A_update, B_update) in kfacs.items():
            A, B = target[layer_idx]
            exponential_moving_average(A, A_update, ema_factor)
            exponential_moving_average(B, B_update, ema_factor)

    def _update_inv_kfac(self):
        """Update the KFAC inverse that is used as pre-conditioner.

        In-place updates the `kfac_preconditioner_inv` attribute.
        """
        damping = self._group["damping"]

        for layer_idx in self.layer_idxs:
            A2, A1 = self.kfacs_interior[layer_idx]
            B2, B1 = self.kfacs_boundary[layer_idx]

            A2, A1 = KFAC.add_damping(A2, A1, damping, "same")
            B2, B1 = KFAC.add_damping(B2, B1, damping, "same")

            self.kfac_preconditioner_inv[layer_idx] = InverseKroneckerSum(  # noqa: B909
                A1, A2, B1, B2, inv_dtype=self.kfac_preconditioner_inv_dtype
            )

    @no_grad()
    def _apply_inv_kfac(self, v: Tensor) -> Tensor:
        """Multiply the KFAC pre-conditioner onto a vector.

        Args:
            v: A vector to multiply the KFAC pre-conditioner with.

        Returns:
            The product of the inverse damped KFAC with the vector.
        """
        # split into parameters
        v_list = [
            v_p.reshape_as(p)
            for v_p, p in zip(v.split([p.numel() for p in self.params]), self.params)
        ]

        kfac_v_list = []

        # multiply with the transpose Jacobian
        for i, layer_idx in enumerate(self.layer_idxs):
            # combine weight and bias
            v_idx = cat([v_list[2 * i], v_list[2 * i + 1].unsqueeze(-1)], dim=1)
            kfac_v_idx = self.kfac_preconditioner_inv[layer_idx] @ v_idx
            # un-combine weight and bias
            kfac_v_list.extend([kfac_v_idx[:, :-1], kfac_v_idx[:, -1]])

        # flatten and concatenate
        return cat([kfac_v.flatten() for kfac_v in kfac_v_list])
