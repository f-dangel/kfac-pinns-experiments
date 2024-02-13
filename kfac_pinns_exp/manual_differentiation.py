"""Functionality to manually compute derivatives of sequential neural networks."""

from typing import List, Optional

from einops import einsum
from torch import Tensor, arange, ones_like, zeros, zeros_like
from torch.nn import Linear, Module, ReLU, Sigmoid, Tanh


def manual_forward(layers: List[Module], x: Tensor) -> List[Tensor]:
    """Apply a sequence of layers to an input.

    Args:
        layers: A list of layers.
        x: The input to the first layer. First dimension is batch dimension.

    Returns:
        A list of intermediate representations. First entry is the original input, last
        entry is the last layer's output.

    Raises:
        ValueError: If a layer uses in-place operations.
    """
    activations = [x]

    for layer in layers:
        if getattr(layer, "inplace", False):
            raise ValueError(
                f"Layers with in-place operations are not supported. Got {layer}."
            )

        x = layer(x)
        activations.append(x)

    return activations


def manual_backward(
    layers: List[Module],
    activations: List[Tensor],
    grad_outputs: Optional[Tensor] = None,
) -> List[Tensor]:
    """Backpropagate through all layers, computing gradients w.r.t. activations.

    Args:
        layers: A list of layers.
        activations: A list of intermediate representations from the forward pass.
            First entry is the original input, last entry is the last layer's output.
        grad_outputs: The vector that is backpropagated through the layers. Must have
            same size as the last element of `activations`. If not specified, a vector
            of ones will be used.

    Returns:
        A list of gradients. First entry is the gradient of the output w.r.t. the input,
        last entry is the gradient of the output w.r.t. the last layer's output, etc.

    Raises:
        ValueError: If the number of layers and activations do not match.
        ValueError: If `grad_output` is specified but has incorrect size.
    """
    if len(layers) != len(activations) - 1:
        raise ValueError(
            f"Number of layers ({len(layers)}) must equal number of activations - 1. "
            + f"got {len(activations)} activations."
        )

    # set up empty gradient buffers
    gradients = [zeros_like(a) for a in activations[:-1]] + [
        ones_like(activations[-1]) if grad_outputs is None else grad_outputs
    ]

    # backpropagate, starting from the last layer
    for i in range(len(layers), 0, -1):
        layer = layers[i - 1]
        inputs, outputs = activations[i - 1], activations[i]
        grad_outputs = gradients[i]
        gradients[i - 1] += manual_backward_layer(layer, inputs, outputs, grad_outputs)

    return gradients


def manual_backward_layer(
    layer: Module, inputs: Tensor, outputs, grad_outputs: Tensor
) -> Tensor:
    """Backpropagate through a layer (output to input).

    Args:
        layer: The layer to backpropagate through.
        inputs: The input to the layer from the forward pass.
        outputs: The output of the layer from the forward pass.
        grad_outputs: The gradient of the loss w.r.t. the layer's output.

    Returns:
        The gradient of the loss w.r.t. the layer's input. Has same
        shape as the layer's input.

    Raises:
        NotImplementedError: If manual backpropagation for a layer is not implemented.
        RuntimeError: If `grad_outputs` or the return value have incorrect shapes.
    """
    if grad_outputs.shape != outputs.shape:
        raise RuntimeError(
            "Grad output must have same shape as output. "
            + f"Got {grad_outputs.shape} and {outputs.shape}."
        )

    if isinstance(layer, Linear):
        # ... denotes an arbitrary number of additional dimensions, e.g. sequence length
        grad_inputs = einsum(
            layer.weight, grad_outputs, "d_out d_in, batch ... d_out -> batch ... d_in"
        )
    elif isinstance(layer, Sigmoid):
        # σ' = σ(1 - σ)
        grad_inputs = grad_outputs * outputs * (1 - outputs)
    elif isinstance(layer, ReLU):
        # ReLU' = 1 if x > 0, 0 otherwise
        grad_inputs = grad_outputs * (outputs > 0).float()
    elif isinstance(layer, Tanh):
        # σ' = 1 - σ^2
        grad_inputs = grad_outputs * (1 - outputs**2)
    else:
        raise NotImplementedError(f"Backpropagation through {layer} not implemented.")

    if grad_inputs.shape != inputs.shape:
        raise RuntimeError(
            "Grad inputs must have same shape as inputs. "
            + f"Got {grad_inputs.shape} and {inputs.shape}."
        )

    return grad_inputs


def manual_hessian_backward(
    layers: List[Module],
    activations: List[Tensor],
    gradients: List[Tensor],
    hess_outputs: Optional[Tensor] = None,
) -> List[Tensor]:
    """Hessian-backpropagate through all layers, computing Hessians w.r.t. activations.

    Args:
        layers: A list of layers.
        activations: A list of intermediate representations from the forward pass.
            First entry is the original input, last entry is the last layer's output.
            The last entry must have shape `[batch_size, 1]` because we only support
            computing Hessians of scalar-valued functions.
        gradients: A list of gradients. First entry is the gradient of the output w.r.t.
            the input, last is the gradient of the output w.r.t. the last layer's
            output, etc.
        hess_outputs: Hessian w.r.t. the last output that will be backpropagated. Has
            shape `[batch_size, *last.shape[1:], *last.shape[1:]]` where `last` is the
            last entry of `activations`. If unspecified, a zero tensor will be used.

    Returns:
        A list of Hessians. First entry is the backpropagated Hessian to the input,
        last entry is the Hessian of the output w.r.t. the last layer's output, etc.
        For an activation of shape `[batch_size, *]`, the corresponding Hessian has
        shape `[batch_size, *, *]`.

    Raises:
        ValueError: If the number of layers and activations or the number of layers
            and gradients do not match.
        ValueError: If the output of the last layer is not a batched scalar.
    """
    if len(layers) != len(activations) - 1:
        raise ValueError(
            f"Number of layers ({len(layers)}) must equal number of activations - 1. "
            + f"got {len(activations)} activations."
        )
    if len(layers) != len(gradients) - 1:
        raise ValueError(
            f"Number of layers ({len(layers)}) must equal number of gradients - 1. "
            + f"got {len(gradients)} gradients."
        )
    last = activations[-1]
    if last.ndim != 2 or last.shape[1] != 1:
        raise ValueError(
            f"Last activation must be a batched scalar. Got shape {last.shape}."
        )

    # set up empty Hessian buffers
    batch_size = last.shape[0]
    hessians: List[Tensor] = []
    for act in activations:
        hess_act_shape = (batch_size, *act.shape[1:], *act.shape[1:])
        hessians.append(zeros(*hess_act_shape, device=act.device, dtype=act.dtype))
    hessians.append(
        hess_outputs
        if hess_outputs is not None
        else zeros(
            batch_size,
            *last.shape[1:],
            *last.shape[1:],
            device=last.device,
            dtype=last.dtype,
        )
    )

    # backpropagate, starting from the last layer
    for i in range(len(layers), 0, -1):
        layer = layers[i - 1]
        inputs, outputs = activations[i - 1], activations[i]
        grad_outputs = gradients[i]
        hess_outputs = hessians[i]
        hessians[i - 1] += manual_hessian_backward_layer(
            layer, inputs, outputs, grad_outputs, hess_outputs
        )

    return hessians


def manual_hessian_backward_layer(
    layer: Module,
    inputs: Tensor,
    outputs: Tensor,
    grad_outputs: Tensor,
    hess_outputs: Tensor,
) -> Tensor:
    """Hessian-backpropagate through a layer.

    Args:
        layer: The layer to backpropagate through.
        inputs: The input to the layer from the forward pass.
        outputs: The output of the layer from the forward pass.
        grad_outputs: The gradient of the loss w.r.t. the layer's output.
        hess_outputs: The Hessian of the loss w.r.t. the layer's output.

    Returns:
        The Hessian of the loss w.r.t. the layer's input. If the input has shape
        `[batch_size, *]`, the Hessian has shape `[batch_size, *, *]`.

    Raises:
        NotImplementedError: If manual Hessian backpropagation for a layer is not
            implemented.
        RuntimeError: If the incoming or outgoing Hessians have incorrect shapes.
    """
    batch_size = inputs.shape[0]
    hess_inputs_shape = (batch_size, *inputs.shape[1:], *inputs.shape[1:])

    hess_outputs_shape = (batch_size, *outputs.shape[1:], *outputs.shape[1:])
    if hess_outputs.shape != hess_outputs_shape:
        raise RuntimeError(
            "Hess outputs has incorrect shape. "
            + f"Expected {hess_outputs_shape}. Got {hess_outputs.shape}."
        )

    if isinstance(layer, Linear):
        shared = outputs.shape[1:-1].numel()
        out_features = outputs.shape[-1]
        hess_outputs_flat = hess_outputs.reshape(
            batch_size, shared, out_features, shared, out_features
        )
        hess_inputs_flat = einsum(
            layer.weight,
            hess_outputs_flat,
            layer.weight,
            "d_out1 d_in1, batch shared1 d_out1 shared2 d_out2, d_out2 d_in2 -> "
            + "batch shared1 d_in1 shared2 d_in2",
        )
        hess_inputs = hess_inputs_flat.reshape(hess_inputs_shape)

    elif isinstance(layer, Sigmoid):
        num_features = inputs.shape[1:].numel()

        hess_outputs_flat = hess_outputs.reshape(batch_size, num_features, num_features)
        outputs_flat = outputs.flatten(start_dim=1)
        grad_outputs_flat = grad_outputs.flatten(start_dim=1)

        # backpropagate incoming curvature
        # σ' = σ(1 - σ)
        d_sigma_flat = outputs_flat * (1 - outputs_flat)
        hess_inputs_flat = einsum(
            d_sigma_flat,
            hess_outputs_flat,
            d_sigma_flat,
            "batch d1, batch d1 d2, batch d2 -> batch d1 d2",
        )

        # add curvature from layer
        # σ'' = σ(1 - σ)(1 - 2σ)
        d2_sigma_flat = d_sigma_flat * (1 - 2 * outputs_flat)
        local_curvature_flat_diag = d2_sigma_flat * grad_outputs_flat
        # add to the diagonal
        idxs = arange(num_features, device=outputs_flat.device)
        hess_inputs_flat[:, idxs, idxs] += local_curvature_flat_diag

        hess_inputs = hess_inputs_flat.reshape(hess_inputs_shape)
    elif isinstance(layer, ReLU):
        num_features = inputs.shape[1:].numel()

        hess_outputs_flat = hess_outputs.reshape(batch_size, num_features, num_features)
        outputs_flat = outputs.flatten(start_dim=1)
        grad_outputs_flat = grad_outputs.flatten(start_dim=1)

        # backpropagate incoming curvature
        d_relu_flat = (outputs_flat > 0).to(outputs_flat.dtype)
        hess_inputs_flat = einsum(
            d_relu_flat,
            hess_outputs_flat,
            d_relu_flat,
            "batch d1, batch d1 d2, batch d2 -> batch d1 d2",
        )

        # NOTE no local curvature is added by ReLU
        hess_inputs = hess_inputs_flat.reshape(hess_inputs_shape)
    elif isinstance(layer, Tanh):
        num_features = inputs.shape[1:].numel()

        hess_outputs_flat = hess_outputs.reshape(batch_size, num_features, num_features)
        outputs_flat = outputs.flatten(start_dim=1)
        grad_outputs_flat = grad_outputs.flatten(start_dim=1)

        # backpropagate incoming curvature
        # σ' = 1 - σ^2
        d_sigma_flat = 1 - outputs_flat**2
        hess_inputs_flat = einsum(
            d_sigma_flat,
            hess_outputs_flat,
            d_sigma_flat,
            "batch d1, batch d1 d2, batch d2 -> batch d1 d2",
        )

        # add curvature from layer
        # σ'' = -2 * σ (1 - σ^2)
        d2_sigma_flat = -2 * outputs_flat * d_sigma_flat
        local_curvature_flat_diag = d2_sigma_flat * grad_outputs_flat
        # add to the diagonal
        idxs = arange(num_features, device=outputs_flat.device)
        hess_inputs_flat[:, idxs, idxs] += local_curvature_flat_diag

        hess_inputs = hess_inputs_flat.reshape(hess_inputs_shape)
    else:
        raise NotImplementedError(
            f"Hessian backpropagation through {layer} not implemented."
        )

    if hess_inputs.shape != hess_inputs_shape:
        raise RuntimeError(
            "Hess inputs has incorrect shape. "
            + f"Expected {hess_inputs_shape}. Got {hess_inputs.shape}."
        )

    return hess_inputs
