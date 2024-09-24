"""Implements the SPRING optimizer (https://arxiv.org/pdf/2401.10190v1) for PINNs."""

from math import sqrt
from typing import Dict, List, Tuple

from einops import einsum
from torch import Tensor, cat, zeros
from torch.nn import Module

from kfac_pinns_exp import (
    fokker_planck_isotropic_equation,
    heat_equation,
    log_fokker_planck_isotropic_equation,
    poisson_equation,
)
from kfac_pinns_exp.pinn_utils import (
    evaluate_boundary_loss_with_layer_inputs_and_grad_outputs,
)

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
    Dict[int, Tensor],
    Dict[int, Tensor],
    Dict[int, Tensor],
    Dict[int, Tensor],
]:
    """Evaluate the interior and boundary losses, and the layer inputs/grad outputs.

    Args:
        layers: The layers that form the neural network.
        X_Omega: The input data for the interior loss.
        y_Omega: The target data for the interior loss.
        X_dOmega: The input data for the boundary loss.
        y_dOmega: The target data for the boundary loss.
        ggn_type: The GGN type.
        approximation: The Gramian approximation.

    Returns:
        The differentiable interior loss, differentiable boundary loss,
        layer inputs of the interior loss, layer gradient outputs of the interior loss,
        layer inputs of the boundary loss, layer gradient outputs of the boundary loss.
    """
    assert ggn_type == "type-2"
    interior_evaluator = EVAL_FNS[equation]["interior"]
    boundary_evaluator = EVAL_FNS[equation]["boundary"]

    interior_loss, interior_inputs, interior_grad_outputs = interior_evaluator(
        layers, X_Omega, y_Omega, ggn_type
    )
    boundary_loss, boundary_inputs, boundary_grad_outputs = boundary_evaluator(
        layers, X_dOmega, y_dOmega, ggn_type
    )

    return (
        interior_loss,
        boundary_loss,
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
        JTv.extend([JTv_weight, JTv_bias])

    # grad_outputs are scaled by 1/N, but we need 1/√N for the transposed Jacobian
    (N,) = {t.shape[0] for t in list(inputs.values()) + list(grad_outputs.values())}
    for v in JTv:
        v.mul_(sqrt(N))

    return JTv
