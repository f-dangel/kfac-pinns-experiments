"""Implements linear operators."""

from math import sqrt
from typing import List

from einops import einsum
from torch import Tensor, cat, zeros
from torch.autograd import grad
from torch.nn import Linear, Module

from kfac_pinns_exp import heat_equation, poisson_equation
from kfac_pinns_exp.poisson_equation import get_backpropagated_error
from kfac_pinns_exp.utils import bias_augmentation


class BoundaryGramianLinearOperator:
    """Linear operator for the boundary Gramian.

    Pre-computes the information required to multiply with the Gramian in one
    backward pass. This saves computation when doing multiple Gramian-vector
    products, compared to matrix-free multiplication with the Gramian based on
    nested autodiff.

    Attributes:
        SUPPORTED_APPROXIMATIONS: The supported Gramian approximations.
    """

    SUPPORTED_APPROXIMATIONS = {"full", "per_layer"}

    def __init__(
        self,
        equation: str,
        layers: List[Module],
        X: Tensor,
        y: Tensor,
        ggn_type: str = "type-2",
        approximation: str = "full",
    ):
        """Pre-compute the information for the Gramian-vector product.

        Args:
            equation: The type of equation to solve.
            layers: The neural network's layers.
            X: The input data tensor.
            y: The target data tensor.
            ggn_type: The type of GGN to use.
            approximation: The Gramian approximation. Can be `'full'` or `'per_layer'`.

        Raises:
            NotImplementedError: If there are trainable parameters in unsupported
                layers.
            ValueError: For unsupported values of `approximation`.
        """
        self.layers = layers
        self.batch_size = X.shape[0]

        if approximation not in self.SUPPORTED_APPROXIMATIONS:
            raise ValueError(
                f"Approximation '{approximation}' not supported. "
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
        loss_evaluator = {
            "poisson": poisson_equation.evaluate_boundary_loss,
            "heat": heat_equation.evaluate_boundary_loss,
        }[equation]
        loss, residual, intermediates = loss_evaluator(layers, X, y)

        # collect all layer output gradients
        layer_outputs = [intermediates[idx + 1] for idx in self.layer_idxs]
        error = get_backpropagated_error(residual, ggn_type).mul_(sqrt(self.batch_size))
        grad_outputs = grad(
            residual, layer_outputs, grad_outputs=error, retain_graph=True
        )
        self.layer_grad_outputs = {
            idx: g for g, idx in zip(grad_outputs, self.layer_idxs)
        }

        # collect all layer inputs
        self.layer_inputs = {
            idx: bias_augmentation(intermediates[idx], 1) for idx in self.layer_idxs
        }

        # compute the Euclidean gradient
        self.grad = grad(loss, self.params)

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
            v: The vector to multiply with the Gramian in list format.

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
