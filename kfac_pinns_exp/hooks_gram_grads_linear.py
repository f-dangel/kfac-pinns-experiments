"""Contains hooks to compute the Gramian of a linear layer during a backward pass."""

from einops import einsum
from torch import Tensor


def from_output(
    grad_output: Tensor, layer_input: Tensor, param_name: str, accumulator: Tensor
) -> None:
    """Backward hook which computes the Gram gradients from the forward pass.

    Should be installed on the output a of a linear layer.

    Args:
        grad_output: Gradient of the Laplacian w.r.t. the layer's output.
        layer_input: Layer's input.
        param_name: W.r.t. which parameter the Gram gradients should be computed.
            Must be `'weight'` or `'bias`.
        accumulator: Tensor to accumulate the Gram gradients in.

    Raises:
        ValueError: If the parameter name is invalid.
    """
    if param_name not in {"bias", "weight"}:
        raise ValueError("Param name must be 'weight' or 'bias'. Got {param_name}.")

    if param_name == "bias":
        accumulator.add_(grad_output.detach())
    else:
        accumulator.add_(
            einsum(
                grad_output.detach(),
                layer_input.detach(),
                "batch d_out, batch d_in -> batch d_out d_in",
            )
        )


def from_grad_input(
    grad_grad_input: Tensor, layer_grad_output: Tensor, accumulator: Tensor
) -> None:
    """Backward hook which computes the Gram gradients from the backward pass.

    Modifies `accumulator`.

    Args:
        grad_grad_input: Gradient of the Laplacian w.r.t. the neural network's gradient
            w.r.t. the layer input.
        layer_grad_output: Gradient of the neural network w.r.t. the layer's output.
        accumulator: Tensor to accumulate the Gram gradients in.
    """
    accumulator.add_(
        einsum(
            layer_grad_output.detach(),
            grad_grad_input.detach(),
            "batch d_out, batch d_in -> batch d_out d_in",
        )
    )


def from_hess_input(
    grad_hess_input: Tensor,
    layer_hess_output: Tensor,
    weight: Tensor,
    accumulator: Tensor,
) -> None:
    """Backward hook computing Gram gradients from the Hessian backward pass.

    Modifies `accumulator`.

    Args:
        grad_hess_input: Gradient of the Laplacian w.r.t. the neural network's Hessian
            w.r.t. the layer input.
        layer_hess_output: Hessian of the neural network w.r.t. the layer's output.
        weight: The layer's weight.
        accumulator: Tensor to accumulate the Gram gradients in.
    """
    accumulator.add_(
        einsum(
            grad_hess_input.detach(),
            weight.detach(),
            layer_hess_output.detach(),
            "batch d_in1 d_in2, d_out1 d_in1, batch d_out1 d_out2 "
            + "-> batch d_out2 d_in2",
        ),
        alpha=2.0,
    )
