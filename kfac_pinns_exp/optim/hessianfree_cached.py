"""Argument parser and implementation of Hessian-free optimizer+cached linear operators.

The implementation is at https://github.com/ltatzel/PyTorchHessianFree.
"""

from argparse import ArgumentParser, Namespace
from typing import Callable, List, Tuple

from hessianfree.optimizer import HessianFree
from torch import Tensor, cat, no_grad
from torch.nn import Linear, Module

from kfac_pinns_exp.linops import GramianLinearOperator
from kfac_pinns_exp.parse_utils import parse_known_args_and_remove_from_argv


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
    args = parse_known_args_and_remove_from_argv(parser)

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
    SUPPORTED_EQUATIONS = {"poisson", "heat"}

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

        layer_idxs = []
        for idx, layer in enumerate(layers):
            if isinstance(layer, Linear):
                if (
                    layer.weight.requires_grad
                    and layer.bias is not None
                    and layer.bias.requires_grad
                ):
                    layer_idxs.append(idx)
                elif any(p.requires_grad for p in layer.parameters()):
                    raise NotImplementedError(
                        "Trainable linear layers must have differentiable weight+bias."
                    )
            elif any(p.requires_grad for p in layer.parameters()):
                raise NotImplementedError(
                    "Trainable parameters must be in linear layers."
                )
        params = sum((list(layers[idx].parameters()) for idx in layer_idxs), [])

        super().__init__(
            params,
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

        self.layers = layers
        self.equation = equation
        self.approximation = approximation

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
                (g_int + g_bnd).flatten()
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

        return super().step(
            forward,
            grad=grad,
            mvp=mvp,
            M_func=None,
            test_deterministic=test_deterministic,
        )
