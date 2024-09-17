"""Implements linear operators."""

from math import sqrt
from typing import List

from einops import einsum
from torch import Tensor, cat, zeros
from torch.autograd import grad
from torch.nn import Linear, Module

from kfac_pinns_exp import (
    fokker_planck_isotropic_equation,
    heat_equation,
    log_fokker_planck_isotropic_equation,
    poisson_equation,
)
from kfac_pinns_exp.pinn_utils import (
    evaluate_boundary_loss_with_layer_inputs_and_grad_outputs,
)


class GramianLinearOperator:
    """Class for linear operators representing a Gramian.

    Pre-computes the information required to multiply with the Gramian in one
    backward pass. This saves computation when doing multiple Gramian-vector
    products, compared to matrix-free multiplication with the Gramian based on
    nested autodiff.

    Attributes:
        SUPPORTED_APPROXIMATIONS: The supported Gramian approximations.
        SUPPORTED_GGN_TYPES: The supported GGN types.
        SUPPORTED_LOSS_TYPES: The supported loss types.
        EVAL_FNS: The functions to evaluate the loss, inputs and output gradients
            for each equation type.
    """

    SUPPORTED_APPROXIMATIONS = {"full", "per_layer"}
    SUPPORTED_GGN_TYPES = {"type-2"}
    SUPPORTED_LOSS_TYPES = {"interior", "boundary"}
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

    def __init__(
        self,
        equation: str,
        layers: List[Module],
        X: Tensor,
        y: Tensor,
        loss_type: str,
        ggn_type: str = "type-2",
        approximation: str = "full",
    ):
        """Pre-compute the information for the Gramian-vector product.

        Args:
            equation: The type of equation to solve.
            layers: The neural network's layers.
            X: The input data tensor.
            y: The target data tensor.
            loss_type: The type of loss to use. Can be `'interior'` or `'boundary'`.
            ggn_type: The type of GGN to use. Default: `'type-2'`.
            approximation: The Gramian approximation. Can be `'full'` or `'per_layer'`.

        Raises:
            NotImplementedError: If there are trainable parameters in unsupported
                layers.
            ValueError: For unsupported values of `approximation`.
            ValueError: For unsupported values of `ggn_type`.
        """
        self.layers = layers
        self.batch_size = X.shape[0]
        self.ggn_type = ggn_type
        self.equation = equation
        self.loss_type = loss_type

        if ggn_type not in self.SUPPORTED_GGN_TYPES:
            raise ValueError(
                f"GGN type {ggn_type!r} not supported. "
                f"Choose from {self.SUPPORTED_GGN_TYPES}."
            )

        if approximation not in self.SUPPORTED_APPROXIMATIONS:
            raise ValueError(
                f"Approximation {approximation!r} not supported. "
                f"Choose from {self.SUPPORTED_APPROXIMATIONS}."
            )
        self.approximation = approximation

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
        # compute quantities required for Gramian-vector products
        eval_fn = self.EVAL_FNS[equation][loss_type]
        loss, self.layer_inputs, self.layer_grad_outputs = eval_fn(
            layers, X, y, ggn_type
        )

        # `grad_outputs` have scaling `1/N`, but we need `1/sqrt(N)` for the matvec
        batch_size = X.shape[0]
        for g_out in self.layer_grad_outputs.values():
            g_out.mul_(sqrt(batch_size))

        self.grad = grad(loss, self.params, allow_unused=True, materialize_grads=True)

    def __matmul__(self, v: Tensor) -> Tensor:
        """Multiply the Gramian onto a vector.

        Args:
            v: The vector to multiply with the Gramian. Has shape `[D]` where `D` is
                the total number of parameters in the network.

        Returns:
            The result of the Gramian-vector product. Has shape `[D]`.
        """
        # split into parameters
        v_list = [
            v_p.reshape_as(p)
            for v_p, p in zip(v.split([p.numel() for p in self.params]), self.params)
        ]

        # matrix-vector product in list format
        matmul_func = {"full": self._matmul_full, "per_layer": self._matmul_per_layer}[
            self.approximation
        ]
        Gv_list = matmul_func(v_list)

        # flatten and concatenate
        return cat([Gv.flatten() for Gv in Gv_list])

    def _matmul_full(self, v_list: List[Tensor]) -> List[Tensor]:
        """Multiply the full Gramian onto a vector.

        Args:
            v_list: The vector to multiply with the Gramian in list format.

        Returns:
            The result of the Gramian-vector product in list format.
        """
        (dev,) = {v.device for v in v_list}
        (dt,) = {v.dtype for v in v_list}
        JT_v = zeros(self.batch_size, device=dev, dtype=dt)

        # multiply with the transpose Jacobian
        for i, layer_idx in enumerate(self.layer_idxs):
            z = self.layer_inputs[layer_idx]
            g = self.layer_grad_outputs[layer_idx]
            # combine weight and bias
            v_idx = cat([v_list[2 * i], v_list[2 * i + 1].unsqueeze(-1)], dim=1)
            JT_v.add_(einsum(z, g, v_idx, "n ... d_in, n ... d_out, d_out d_in -> n"))

        result = []

        # multiply with the Jacobian
        for layer_idx in self.layer_idxs:
            z = self.layer_inputs[layer_idx]
            g = self.layer_grad_outputs[layer_idx]
            v_idx = einsum(z, g, JT_v, "n ... d_in, n ... d_out, n -> d_out d_in")
            # un-combine weight and bias
            result.extend([v_idx[:, :-1], v_idx[:, -1]])

        return result

    def _matmul_per_layer(self, v_list: List[Tensor]) -> List[Tensor]:
        """Multiply the per-layer Gramian onto a vector.

        Args:
            v_list: The vector to multiply with the Gramian in list format.

        Returns:
            The result of the per-layer Gramian-vector product in list format.
        """
        result = []

        # multiply with the per-layer Gramian
        for i, layer_idx in enumerate(self.layer_idxs):
            z = self.layer_inputs[layer_idx]
            g = self.layer_grad_outputs[layer_idx]
            # combine weight and bias
            v_idx = cat([v_list[2 * i], v_list[2 * i + 1].unsqueeze(-1)], dim=1)
            # multiply with JJT
            JT_v = einsum(z, g, v_idx, "n ... d_in, n ... d_out, d_out d_in -> n")
            v_idx = einsum(z, g, JT_v, "n ... d_in, n ... d_out, n -> d_out d_in")
            # un-combine weight and bias
            result.extend([v_idx[:, :-1], v_idx[:, -1]])

        return result
