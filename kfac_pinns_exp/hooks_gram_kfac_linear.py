"""Hooks to compute Kronecker-factorized approximate Gram (KFAG) for `nn.Linear`."""

from einops import einsum
from torch import Tensor, cat


def validate_reduction(reduction: str) -> None:
    """Check that the reduction string has a valid value.

    Args:
        reduction: The reduction string.

    Raises:
        ValueError: If the reduction is invalid.
    """
    allowed = {"sum", "mean"}
    if reduction not in allowed:
        raise ValueError(f"Invalid reduction: {reduction}. Allowed: {allowed}.")


def from_output(
    grad_output: Tensor,
    layer_input: Tensor,
    bias: bool,
    kfag_A_accumulator: Tensor,
    kfag_B_accumulator: Tensor,
    reduction: str = "sum",
) -> None:
    """Backward hook computing KFAG contribution from the forward pass.

    Should be installed on the output a of a linear layer.

    Args:
        grad_output: Gradient of the Laplacian w.r.t. the layer's output.
        layer_input: Layer's input.
        bias: Whether the layer has a bias.
        kfag_A_accumulator: Tensor to accumulate the first KFAG factor in.
        kfag_B_accumulator: Tensor to accumulate the second KFAG factor in.
        reduction: The reduction that was used to accumulate the Laplacian's
            over a batch. Must be one of `'sum'`, `'mean'`. Default: `'sum'`.
    """
    validate_reduction(reduction)

    if layer_input.ndim != 2 or grad_output.ndim != 2:
        raise ValueError("Expected 2D input and output tensors.")

    batch_size = layer_input.shape[0]

    # append ones to layer input if it has a bias
    if bias:
        inputs = cat(
            [layer_input, layer_input.new_ones((batch_size, 1))],
            dim=1,
        )
    else:
        inputs = layer_input

    kfag_A_accumulator.add_(
        einsum(inputs, inputs, "batch i, batch j -> i j"),
        alpha=1.0 / batch_size if reduction == "mean" else 1.0,
    )
    kfag_B_accumulator.add_(
        einsum(grad_output, grad_output, "batch i, batch j -> i j"),
        alpha=batch_size**2 if reduction == "mean" else 1.0,
    )


def from_grad_input(
    grad_grad_input: Tensor,
    layer_grad_output: Tensor,
    bias: bool,
    kfag_A_accumulator: Tensor,
    kfag_B_accumulator: Tensor,
    reduction: str = "sum",
) -> None:
    """Backward hook computing KFAG contribution from the backward pass.

    Should be installed on the input gradient of a linear layer.

    Args:
        grad_grad_input: Gradient of the Laplacian w.r.t. the neural network's gradient
            w.r.t. the layer input.
        layer_grad_output: Gradient of the neural network w.r.t. the layer's output.
        bias: Whether the layer has a bias.
        kfag_A_accumulator: Tensor to accumulate the first KFAG factor in.
        kfag_B_accumulator: Tensor to accumulate the second KFAG factor in.
        reduction: The reduction that was used to accumulate the Laplacian's
            over a batch. Must be one of `'sum'`, `'mean'`. Default: `'sum'`.
    """
    validate_reduction(reduction)

    if layer_grad_output.ndim != 2 or grad_grad_input.ndim != 2:
        raise ValueError("Expected 2D input and output tensors.")

    batch_size = layer_grad_output.shape[0]

    # append zeros to neural network gradients if layer has a bias
    if bias:
        grad_grad_in = cat(
            [grad_grad_input, grad_grad_input.new_zeros((batch_size, 1))],
            dim=1,
        )
    else:
        grad_grad_in = grad_grad_input

    kfag_A_accumulator.add_(
        einsum(grad_grad_in, grad_grad_in, "batch i, batch j -> i j"),
        alpha=batch_size if reduction == "mean" else 1.0,
    )
    kfag_B_accumulator.add_(
        einsum(layer_grad_output, layer_grad_output, "batch i, batch j -> i j"),
        alpha=batch_size**2 if reduction == "mean" else 1.0,
    )


def from_hess_input(
    grad_hess_input: Tensor,
    layer_hess_output: Tensor,
    weight: Tensor,
    bias: bool,
    kfag_A_accumulator: Tensor,
    kfag_B_accumulator: Tensor,
    reduction: str = "sum",
) -> None:
    """Backward hook computing Gram gradients from the Hessian backward pass.

    Modifies `accumulator`.

    Args:
        grad_hess_input: Gradient of the Laplacian w.r.t. the neural network's Hessian
            w.r.t. the layer input.
        layer_hess_output: Hessian of the neural network w.r.t. the layer's output.
        weight: The layer's weight.
        bias: Whether the layer has a bias.
        kfag_A_accumulator: Tensor to accumulate the first KFAG factor in.
        kfag_B_accumulator: Tensor to accumulate the second KFAG factor in.
        reduction: The reduction that was used to accumulate the Laplacian's
            over a batch. Must be one of `'sum'`, `'mean'`. Default: `'sum'`.
    """
    validate_reduction(reduction)

    if layer_hess_output.ndim != 3 or grad_hess_input.ndim != 3:
        raise ValueError("Expected 3D input and output tensors.")

    batch_size = grad_hess_input.shape[0]

    # # append zeros to neural network Hessians if layer has a bias
    # if bias:
    #     grad_grad_in = cat(
    #         [grad_grad_input, grad_grad_input.new_zeros((batch_size, 1))],
    #         dim=1,
    #     )
    # else:
    #     grad_grad_in = grad_grad_input

    # kfag_A_accumulator.add_(
    #     einsum(grad_grad_in, grad_grad_in, "batch i, batch j -> i j"),
    #     alpha=batch_size if reduction == "mean" else 1.0,
    # )

    W_WT = einsum(weight, weight, "out1 in, out2 in -> out1 out2")
    X_W_WT_XT = einsum(
        layer_hess_output,
        W_WT,
        layer_hess_output,
        "batch out1 out2, out2 out3, batch out4 out3 -> out1 out4",
    )
    kfag_B_accumulator.add_(
        X_W_WT_XT,
        alpha=batch_size**2 if reduction == "mean" else 1.0,
    )
