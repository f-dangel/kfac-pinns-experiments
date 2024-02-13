"""Implements the KFAC optimizer."""

from functools import partial
from typing import Dict, List, Tuple

from einops import einsum, rearrange
from torch import Tensor, cat, eye, ones_like, zeros
from torch.autograd import grad
from torch.nn import Linear, Module
from torch.utils.hooks import RemovableHandle

from kfac_pinns_exp.exp08_forward_laplacian.demo_forward_laplacian import (
    manual_forward_laplacian,
)
from kfac_pinns_exp.manual_differentiation import manual_forward


def evaluate_interior_loss_and_kfac_expand(
    layers: List[Module], X: Tensor, y: Tensor
) -> Tuple[Tensor, Dict[int, Tuple[Tensor, Tensor]]]:
    """Evaluate the interior loss and compute its KFAC-expand approximation.

    Args:
        layers: The list of layers in the neural network.
        X: The input data.
        y: The target data.

    Returns:
        The (differentiable) interior loss and a dictionary whose keys are the layer
        indices and whose values are the two Kronecker factors.
    """
    kfacs = check_layers_and_initialize_kfac(layers)

    # We turn X's differentiability on to trigger the grad-output based
    # Kronecker factor computation by calling `grad` w.r.t. `X`. We restore the
    # original value later.
    X_original_requires_grad = X.requires_grad
    X.requires_grad = True

    # Compute the forward Laplacian and all the intermediates
    intermediates = manual_forward_laplacian(layers, X)

    ###############################################################################
    #                    COMPUTE INPUT-BASED KRONECKER FACTORS                    #
    ###############################################################################
    batch_size, shared, d0 = X.shape[0], X.shape[1:-1].numel(), X.shape[-1]
    num_outer_products = batch_size * shared * (d0 + 2)
    for layer_idx, (A, _) in kfacs.items():
        for input_type, bias_augmentation in zip(
            ["forward", "directional_gradients", "laplacian"], [1, 0, 0]
        ):
            x = intermediates[layer_idx][input_type]
            add_input_based_kfac_expand(
                x, num_outer_products, bias_augmentation, dest=A
            )

    ###############################################################################
    #                   COMPUTE OUTPUT-BASED KRONECKER FACTORS                    #
    ###############################################################################
    handles: List[RemovableHandle] = []

    # Install hooks that accumulate the outer product into all B's
    for layer_idx, (_, B) in kfacs.items():
        for x in intermediates[layer_idx + 1].values():
            hook = partial(hook_add_output_based_kfac_expand, dest=B)
            hook_handle = x.register_hook(hook)
            handles.append(hook_handle)

    # We want to use the Laplacian in the loss and be able to differentiate
    # through it again. Therefore, set `retain_graph=True`
    laplacian = intermediates[-1]["laplacian"]
    # trigger the backward hooks
    grad(laplacian, X, grad_outputs=ones_like(laplacian), retain_graph=True)

    # remove the hooks & reset original differentiability
    X.requires_grad = X_original_requires_grad
    for handle in handles:
        handle.remove()

    # compute the interior loss
    return 0.5 * ((laplacian + y) ** 2).mean(), kfacs


def evaluate_boundary_loss_and_kfac_expand(
    layers: List[Module], X: Tensor, y: Tensor
) -> Tuple[Tensor, Dict[int, Tuple[Tensor, Tensor]]]:
    """Evaluate the boundary loss and compute its KFAC-expand approximation.

    Args:
        layers: The list of layers in the neural network.
        X: The input data.
        y: The target data.

    Returns:
        The (differentiable) boundary loss and a dictionary whose keys are the layer
        indices and whose values are the two Kronecker factors.
    """
    kfacs = check_layers_and_initialize_kfac(layers)

    # We turn X's differentiability on to trigger the grad-output based
    # Kronecker factor computation by calling `grad` w.r.t. `X`. We restore the
    # original value later.
    X_original_requires_grad = X.requires_grad
    X.requires_grad = True

    # Compute the NN prediction and all intermediates
    intermediates = manual_forward(layers, X)

    ###############################################################################
    #                    COMPUTE INPUT-BASED KRONECKER FACTORS                    #
    ###############################################################################
    batch_size, shared = X.shape[0], X.shape[1:-1].numel()
    num_outer_products = batch_size * shared
    bias_augmentation = 1
    for layer_idx, (A, _) in kfacs.items():
        x = intermediates[layer_idx]
        add_input_based_kfac_expand(x, num_outer_products, bias_augmentation, dest=A)

    ###############################################################################
    #                   COMPUTE OUTPUT-BASED KRONECKER FACTORS                    #
    ###############################################################################
    handles: List[RemovableHandle] = []

    # Install hooks that accumulate the outer product into all B's
    for layer_idx, (_, B) in kfacs.items():
        hook = partial(hook_add_output_based_kfac_expand, dest=B)
        hook_handle = intermediates[layer_idx + 1].register_hook(hook)
        handles.append(hook_handle)

    # We want to use the prediction in the loss and be able to differentiate
    # through it again. Therefore, set `retain_graph=True`
    output = intermediates[-1]
    # trigger the backward hooks
    grad(output, X, grad_outputs=ones_like(output), retain_graph=True)

    # remove the hooks & reset original differentiability
    X.requires_grad = X_original_requires_grad
    for handle in handles:
        handle.remove()

    # compute the interior loss
    return 0.5 * ((output - y) ** 2).mean(), kfacs


def check_layers_and_initialize_kfac(
    layers: List[Module], initialize_to_identity: bool = False
) -> Dict[int, Tuple[Tensor, Tensor]]:
    """Verify all layers are supported and initialize the KFAC factors.

    Args:
        layers: The list of layers in the neural network.
        initialize_to_identity: Whether to initialize the KFAC factors to the identity
            matrix. If `False`, the factors are initialized to zero. Default is `False`.

    Returns:
        A dictionary whose keys are the layer indices and whose values are the two
        Kronecker factors.

    Raises:
        NotImplementedError: If a layer with parameters is not a linear layer with bias
            and both parameters differentiable.
    """
    kfacs = {}

    for layer_idx, layer in enumerate(layers):
        if list(layer.parameters()) and not isinstance(layer, Linear):
            raise NotImplementedError("Only parameters in linear layers are supported.")
        if isinstance(layer, Linear):
            if layer.bias is None:
                raise NotImplementedError("Only layers with bias are supported.")
            if any(not p.requires_grad for p in layer.parameters()):
                raise NotImplementedError("All parameters must require gradients.")
            weight = layer.weight
            kwargs = {"dtype": weight.dtype, "device": weight.device}
            d_out, d_in = weight.shape
            if initialize_to_identity:
                A = eye(d_in + 1, **kwargs)
                B = eye(d_out, **kwargs)
            else:
                A = zeros(d_in + 1, d_in + 1, **kwargs)
                B = zeros(d_out, d_out, **kwargs)
            kfacs[layer_idx] = (A, B)

    return kfacs


def hook_add_output_based_kfac_expand(grad_t: Tensor, dest: Tensor) -> None:
    """Add the gradient's outer product into the destination tensor.

    Args:
        grad_t: The gradient w.r.t. the tensor `t` onto which this hook is installed.
        dest: The destination tensor onto which the gradient outer product is added
            in-place.
    """
    # flatten shared axes into one and use detached tensor for KFAC factor
    grad_t = rearrange(grad_t, "... d_in -> ... d_in").detach()
    dest.add_(einsum(grad_t, grad_t, "n i, n j -> i j"))


def add_input_based_kfac_expand(
    t: Tensor, num_outer_products: int, bias_augmentation: int, dest: Tensor
) -> None:
    """Add the input's outer product into the destination tensor.

    Args:
        t: The input tensor.
        num_outer_products: The number of total outer products to average over.
        bias_augmentation: The augmentation to apply to the input tensor to account for
            the bias. Must be either `0` or `1`.
        dest: The destination tensor onto which the input outer product is added.
    """
    # use detached tensors so that the KFAC factor is not part of the graph
    # flatten batch and shared axes into one
    t = rearrange(t.detach(), "... d_in -> (...) d_in")

    # augment to account for bias
    augmentation_fn = {0: t.new_zeros, 1: t.new_ones}[bias_augmentation]
    t = cat([t, augmentation_fn((t.shape[0], 1))], dim=-1)

    # add outer product to Kronecker factor
    dest.add_(einsum(t, t, "n i, n j -> i j"), alpha=1.0 / num_outer_products)
