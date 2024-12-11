"""Implements the SPRING optimizer (https://arxiv.org/pdf/2401.10190v1) for PINNs."""

from argparse import ArgumentParser, Namespace
from math import sqrt
from typing import Dict, List, Tuple

from einops import einsum
from torch import Tensor, arange, cat, cholesky_solve, zeros, zeros_like
from torch.linalg import cholesky
from torch.nn import Module
from torch.optim import Optimizer

from kfac_pinns_exp import (
    fokker_planck_isotropic_equation,
    heat_equation,
    log_fokker_planck_isotropic_equation,
    poisson_equation,
)
from kfac_pinns_exp.optim.spring_standalone import SPRING as SPRING_STANDALONE
from kfac_pinns_exp.parse_utils import parse_known_args_and_remove_from_argv
from kfac_pinns_exp.pinn_utils import (
    evaluate_boundary_loss_with_layer_inputs_and_grad_outputs,
)

INTERIOR_LOSS_EVALUATORS = {
    "poisson": poisson_equation.evaluate_interior_loss,
    "heat": heat_equation.evaluate_interior_loss,
    "fokker-planck-isotropic": fokker_planck_isotropic_equation.evaluate_interior_loss,
    "log-fokker-planck-isotropic": log_fokker_planck_isotropic_equation.evaluate_interior_loss,  # noqa: B950
}

EVAL_FNS = {
    "poisson": {
        "interior": poisson_equation.evaluate_interior_loss_with_layer_inputs_and_grad_outputs,  # noqa: B950
        "boundary": evaluate_boundary_loss_with_layer_inputs_and_grad_outputs,
    },
    "heat": {
        "interior": heat_equation.evaluate_interior_loss_with_layer_inputs_and_grad_outputs,  # noqa: B950
        "boundary": evaluate_boundary_loss_with_layer_inputs_and_grad_outputs,
    },
    "fokker-planck-isotropic": {
        "interior": fokker_planck_isotropic_equation.evaluate_interior_loss_with_layer_inputs_and_grad_outputs,  # noqa: B950
        "boundary": evaluate_boundary_loss_with_layer_inputs_and_grad_outputs,
    },
    "log-fokker-planck-isotropic": {
        "interior": log_fokker_planck_isotropic_equation.evaluate_interior_loss_with_layer_inputs_and_grad_outputs,  # noqa: B950
        "boundary": evaluate_boundary_loss_with_layer_inputs_and_grad_outputs,
    },
}


def parse_SPRING_args(verbose: bool = False, prefix="SPRING_") -> Namespace:
    """Parse command-line arguments for `SPRING`.

    Args:
        verbose: Whether to print the parsed arguments. Default: `False`.
        prefix: The prefix for the arguments. Default: `'SPRING_'`.

    Returns:
        A namespace with the parsed arguments.
    """
    parser = ArgumentParser(description="Parse arguments for setting up SPRING.")

    parser.add_argument(
        f"--{prefix}lr", type=float, help="Learning rate for SPRING.", required=True
    )
    parser.add_argument(
        f"--{prefix}damping",
        type=float,
        help="Damping factor for the optimizer.",
        default=1e-3,
    )
    parser.add_argument(
        f"--{prefix}decay_factor",
        type=float,
        help="Decay factor of the previous step.",
        default=0.99,
    )
    parser.add_argument(
        f"--{prefix}norm_constraint",
        type=float,
        help="Norm constraint on the natural gradient.",
        default=1e-3,
    )
    parser.add_argument(
        f"--{prefix}adaptive_damping",
        action="store_true",
        help="Whether to use adaptive damping.",
    )
    parser.add_argument(
        f"--{prefix}equation",
        type=str,
        choices=SPRING.SUPPORTED_EQUATIONS,
        help="The equation to solve.",
        default="poisson",
    )

    args = parse_known_args_and_remove_from_argv(parser)

    if verbose:
        print("Parsed arguments for SPRING: ", args)

    return args


class SPRING(Optimizer):
    """SPRING optimizer for PINN problems.

    See https://arxiv.org/pdf/2401.10190v1 for details.
    """

    SUPPORTED_EQUATIONS = EVAL_FNS.keys()

    def __init__(
        self,
        layers: List[Module],
        lr: float,
        damping: float = 1e-3,
        decay_factor: float = 0.99,
        norm_constraint: float = 1e-3,
        adaptive_damping: bool = False,
        equation: str = "poisson",
    ):
        """Set up the SPRING optimizer.

        Args:
            layers: The layers that form the neural network.
            lr: The learning rate.
            damping: The non-negative damping factor (λ in the paper).
                Default: `1e-3` (taken from Section 4 of the paper).
            decay_factor: The decay factor (μ in the paper). Must be in `[0; 1)`.
                Default: `0.99` (taken from Section 4 of the paper).
            norm_constraint: The positive norm constraint (C in the paper).
                Default: `1e-3` (taken from Section 4 of the paper).
            adaptive_damping: If adapt_damping is True, then the damping parameter is
                adapted according to a trust-region strategy. See Section 4.1 in
                https://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf
            equation: Equation to solve. Currently supports `'poisson'`, `'heat'`, and
                `'fokker-planck-isotropic'`. Default: `'poisson'`.

        Raises:
            ValueError: If the optimizer is used with per-parameter options.
            ValueError: If the equation is not supported.
        """
        defaults = dict(
            lr=lr,
            damping=damping,
            decay_factor=decay_factor,
            norm_constraint=norm_constraint,
        )
        params = sum((list(layer.parameters()) for layer in layers), [])
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("SPRING does not support per-parameter options.")

        if equation not in self.SUPPORTED_EQUATIONS:
            raise ValueError(
                f"Equation {equation} not supported."
                f" Supported are: {self.SUPPORTED_EQUATIONS}."
            )
        self.equation = equation
        self.steps = 0
        self.layers = layers
        self.adaptive_damping = adaptive_damping

        # initialize phi
        (group,) = self.param_groups
        for p in group["params"]:
            self.state[p]["phi"] = zeros_like(p)

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
        (group,) = self.param_groups
        params = group["params"]
        lr = group["lr"]
        damping = group["damping"]
        decay_factor = group["decay_factor"]
        norm_constraint = group["norm_constraint"]

        # compute OOT
        (
            interior_loss,
            boundary_loss,
            interior_residual,
            boundary_residual,
            interior_inputs,
            interior_grad_outputs,
            boundary_inputs,
            boundary_grad_outputs,
        ) = evaluate_losses_with_layer_inputs_and_grad_outputs(
            self.layers, X_Omega, y_Omega, X_dOmega, y_dOmega, self.equation
        )
        OOT = compute_JJT(
            interior_inputs,
            interior_grad_outputs,
            boundary_inputs,
            boundary_grad_outputs,
        ).detach()
        # apply damping
        idx = arange(OOT.shape[0], device=OOT.device)
        OOT[idx, idx] = OOT.diag() + damping

        # update zeta
        # compute the residual
        N_dOmega = X_dOmega.shape[0]
        boundary_residual = boundary_residual.detach() / sqrt(N_dOmega)

        N_Omega = X_Omega.shape[0]
        interior_residual = interior_residual.detach() / sqrt(N_Omega)

        epsilon = -cat([interior_residual, boundary_residual]).flatten()

        O_phi = apply_joint_J(
            interior_inputs,
            interior_grad_outputs,
            boundary_inputs,
            boundary_grad_outputs,
            [self.state[p]["phi"] for p in params],
        )
        zeta: Tensor = epsilon - O_phi.mul_(decay_factor)

        # apply inverse of damped OOT to zeta
        step = cholesky_solve(zeta.unsqueeze(-1), cholesky(OOT)).squeeze(-1)

        # apply OT
        step = apply_joint_JT(
            interior_inputs,
            interior_grad_outputs,
            boundary_inputs,
            boundary_grad_outputs,
            step,
        )

        # update phi
        for p, s in zip(params, step):
            self.state[p]["phi"].mul_(decay_factor).add_(s)

        # compute effective learning rate
        norm_phi = sum([(self.state[p]["phi"] ** 2).sum() for p in params]).sqrt()
        scale = min(lr, (sqrt(norm_constraint) / norm_phi).item())

        # update parameters
        for p in params:
            p.data.add_(self.state[p]["phi"], alpha=scale)

        if self.adaptive_damping:
            # compute quadratic model at previous and new iterate
            q = 0.5 * (zeta**2).sum()
            J_step = apply_joint_J(
                interior_inputs,
                interior_grad_outputs,
                boundary_inputs,
                boundary_grad_outputs,
                step,
            )
            q_new = 0.5 * ((J_step + zeta.squeeze()) ** 2).sum()

            # compute loss at new parameter value
            (
                interior_loss_new,
                boundary_loss_new,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = evaluate_losses_with_layer_inputs_and_grad_outputs(
                self.layers, X_Omega, y_Omega, X_dOmega, y_dOmega, self.equation
            )

            multiplier = SPRING_STANDALONE.damping_multiplier(
                interior_loss_new + boundary_loss_new,
                interior_loss + boundary_loss,
                q,
                q_new,
            )
            group["damping"] *= multiplier

        self.steps += 1

        return interior_loss, boundary_loss


def evaluate_losses_with_layer_inputs_and_grad_outputs(
    layers: List[Module],
    X_Omega: Tensor,
    y_Omega: Tensor,
    X_dOmega: Tensor,
    y_dOmega: Tensor,
    equation: str,
    ggn_type: str = "type-2",
) -> Tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Dict[int, Tensor],
    Dict[int, Tensor],
    Dict[int, Tensor],
    Dict[int, Tensor],
]:
    """Evaluate interior and boundary losses/residuals & layer inputs/grad outputs.

    Args:
        layers: The layers that form the neural network.
        X_Omega: The input data for the interior loss.
        y_Omega: The target data for the interior loss.
        X_dOmega: The input data for the boundary loss.
        y_dOmega: The target data for the boundary loss.
        ggn_type: The GGN type.
        equation: The PDE to solve.

    Returns:
        The differentiable interior loss, differentiable boundary loss, differentiable
        interior residual, differentiable boundary residual,
        layer inputs of the interior loss, layer gradient outputs of the interior loss,
        layer inputs of the boundary loss, layer gradient outputs of the boundary loss.
    """
    assert ggn_type == "type-2"
    interior_evaluator = EVAL_FNS[equation]["interior"]
    boundary_evaluator = EVAL_FNS[equation]["boundary"]

    (
        interior_loss,
        interior_res,
        interior_inputs,
        interior_grad_outputs,
    ) = interior_evaluator(layers, X_Omega, y_Omega, ggn_type)
    (
        boundary_loss,
        boundary_res,
        boundary_inputs,
        boundary_grad_outputs,
    ) = boundary_evaluator(layers, X_dOmega, y_dOmega, ggn_type)

    return (
        interior_loss,
        boundary_loss,
        interior_res,
        boundary_res,
        interior_inputs,
        interior_grad_outputs,
        boundary_inputs,
        boundary_grad_outputs,
    )


def compute_JJT(
    interior_inputs: Dict[int, Tensor],
    interior_grad_outputs: Dict[int, Tensor],
    boundary_inputs: Dict[int, Tensor],
    boundary_grad_outputs: Dict[int, Tensor],
) -> Tensor:
    """Compute the Jacobian outer product.

    Args:
        interior_inputs: The layer inputs for the interior loss.
        interior_grad_outputs: The layer gradient outputs for the interior loss.
        boundary_inputs: The layer inputs for the boundary loss.
        boundary_grad_outputs: The layer gradient outputs for the boundary loss.

    Returns:
        The Jacobian outer product.
    """
    (N_Omega,) = {
        t.shape[0]
        for t in list(interior_inputs.values()) + list(interior_grad_outputs.values())
    }
    (N_dOmega,) = {
        t.shape[0]
        for t in list(boundary_inputs.values()) + list(boundary_grad_outputs.values())
    }
    ((dev, dt),) = {
        (t.device, t.dtype)
        for t in list(interior_inputs.values())
        + list(interior_grad_outputs.values())
        + list(boundary_inputs.values())
        + list(boundary_grad_outputs.values())
    }

    JJT = zeros((N_Omega + N_dOmega, N_Omega + N_dOmega), device=dev, dtype=dt)

    for idx in interior_inputs:
        J_boundary = einsum(
            # gradients are scaled by 1/N, but we need 1/√N for the outer product
            boundary_grad_outputs[idx] * sqrt(N_dOmega),
            boundary_inputs[idx],
            "n d_out, n d_in -> n d_out d_in",
        )
        J_interior = einsum(
            # gradients are scaled by 1/N, but we need 1/√N for the outer product
            interior_grad_outputs[idx] * sqrt(N_Omega),
            interior_inputs[idx],
            "n ... d_out, n ... d_in -> n d_out d_in",
        )
        J = cat([J_interior, J_boundary], dim=0).flatten(start_dim=1)

        JJT.add_(J @ J.T)

    return JJT


def apply_joint_J(
    interior_inputs: Dict[int, Tensor],
    interior_grad_outputs: Dict[int, Tensor],
    boundary_inputs: Dict[int, Tensor],
    boundary_grad_outputs: Dict[int, Tensor],
    v: List[Tensor],
) -> Tensor:
    """Multiply the Jacobian onto a vector in parameter space.

    Considers both the interior and the boundary loss.

    Args:
        interior_inputs: The layer inputs for the interior loss.
        interior_grad_outputs: The layer gradient outputs for the interior loss.
        boundary_inputs: The layer inputs for the boundary loss.
        boundary_grad_outputs: The layer gradient outputs for the boundary loss.
        v: The vector to multiply the Jacobian with.

    Returns:
        The result of multiplying the Jacobian with the vector. Has shape `(N_Omega +
        N_dOmega,)` where `N_Omega` and `N_dOmega` are the interior and boundary loss
        batch sizes.
    """
    J_interior_v = apply_individual_J(interior_inputs, interior_grad_outputs, v)
    J_boundary_v = apply_individual_J(boundary_inputs, boundary_grad_outputs, v)
    return cat([J_interior_v, J_boundary_v])


def apply_individual_J(
    inputs: Dict[int, Tensor], grad_outputs: Dict[int, Tensor], v: List[Tensor]
) -> Tensor:
    """Multiply the Jacobian onto a vector in parameter space (tensor list format).

    Considers only a single loss, i.e. either the interior or the boundary loss.

    Args:
        inputs: A dictionary containing the inputs to layers with parameters.
        grad_outputs: A dictionary containing the gradient outputs of layers with
            parameters.
        v: The vector to multiply the Jacobian with.

    Returns:
        The result of multiplying the Jacobian with the vector. Has shape `(N,)` where
        `N` is the batch size.
    """
    assert 2 * len(inputs) == 2 * len(grad_outputs) == len(v)

    ((N, dev, dt),) = {
        (t.shape[0], t.device, t.dtype)
        for t in list(inputs.values()) + list(grad_outputs.values())
    }
    Jv = zeros(N, device=dev, dtype=dt)

    for idx, layer_idx in enumerate(inputs):
        v_weight, v_bias = v[2 * idx], v[2 * idx + 1]
        v_joint = cat([v_weight, v_bias.unsqueeze(-1)], dim=1)
        Jv.add_(
            einsum(
                grad_outputs[layer_idx],
                v_joint,
                inputs[layer_idx],
                "n ... d_out, d_out d_in, n ... d_in -> n",
            )
        )

    # grad_outputs are scaled by 1/N, but we need 1/√N for the Jacobian
    return Jv.mul_(sqrt(N))


def apply_joint_JT(
    interior_inputs: Dict[int, Tensor],
    interior_grad_outputs: Dict[int, Tensor],
    boundary_inputs: Dict[int, Tensor],
    boundary_grad_outputs: Dict[int, Tensor],
    v: Tensor,
) -> List[Tensor]:
    """Multiply the transpose Jacobian onto a vector in data space.

    Considers both the interior and the boundary loss.

    Args:
        interior_inputs: The layer inputs for the interior loss.
        interior_grad_outputs: The layer gradient outputs for the interior loss.
        boundary_inputs: The layer inputs for the boundary loss.
        boundary_grad_outputs: The layer gradient outputs for the boundary loss.
        v: The vector to multiply the transpose Jacobian with.

    Returns:
        The result of multiplying the transpose Jacobian with the vector. Has same
        format as the parameter space, i.e. is a tensor list.
    """
    # split into interior and boundary terms
    (N_Omega,) = {
        t.shape[0]
        for t in list(interior_inputs.values()) + list(interior_grad_outputs.values())
    }
    (N_dOmega,) = {
        t.shape[0]
        for t in list(boundary_inputs.values()) + list(boundary_grad_outputs.values())
    }
    v_interior, v_boundary = v.split([N_Omega, N_dOmega])

    return [
        JTv_interior.add_(JTv_boundary)
        for JTv_interior, JTv_boundary in zip(
            apply_individual_JT(interior_inputs, interior_grad_outputs, v_interior),
            apply_individual_JT(boundary_inputs, boundary_grad_outputs, v_boundary),
        )
    ]


def apply_individual_JT(
    inputs: Dict[int, Tensor], grad_outputs: Dict[int, Tensor], v: Tensor
) -> List[Tensor]:
    """Multiply the transpose Jacobian onto a vector in data space.

    Considers only a single loss, i.e. either the interior or the boundary loss.

    Args:
        inputs: A dictionary containing the inputs to layers with parameters.
        grad_outputs: A dictionary containing the gradient outputs of layers with
            parameters.
        v: The vector to multiply the transpose Jacobian with.

    Returns:
        The result of multiplying the transpose Jacobian with the vector. Has same
        format as the parameter space, i.e. is a tensor list.
    """
    assert len(inputs) == len(grad_outputs)

    JTv = []

    for layer_idx in inputs:
        JTv_joint = einsum(
            grad_outputs[layer_idx],
            v,
            inputs[layer_idx],
            "n ... d_out, n, n ... d_in -> d_out d_in",
        )
        JTv_weight, JTv_bias = JTv_joint.split(
            [inputs[layer_idx].shape[-1] - 1, 1], dim=1
        )
        JTv.extend([JTv_weight, JTv_bias.squeeze(1)])

    # grad_outputs are scaled by 1/N, but we need 1/√N for the transposed Jacobian
    (N,) = {t.shape[0] for t in list(inputs.values()) + list(grad_outputs.values())}
    sqrt_N = sqrt(N)

    for v in JTv:
        v.mul_(sqrt_N)

    return JTv
