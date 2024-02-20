"""Functionality for solving the Poisson equation."""

from functools import partial
from math import pi
from typing import Dict, List, Tuple, Union

from einops import einsum, rearrange
from torch import Tensor, cat, cos, ones_like, prod, rand, randint, sin
from torch.autograd import grad
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

from kfac_pinns_exp.autodiff_utils import autograd_gramian, autograd_input_hessian
from kfac_pinns_exp.forward_laplacian import manual_forward_laplacian
from kfac_pinns_exp.kfac_utils import check_layers_and_initialize_kfac
from kfac_pinns_exp.manual_differentiation import manual_forward


# TODO Use code from exp02 once it is merged
def square_boundary(N: int, dim: int) -> Tensor:
    """Returns quadrature points on the boundary of a square.

    Args:
        N: Number of quadrature points.
        dim: Dimension of the Square.

    Returns:
        A tensor of shape (N, dim) that consists of uniformly drawn
        quadrature points.
    """
    X = rand(N, dim)

    dimensions = randint(0, dim, (N,))
    sides = randint(0, 2, (N,))

    for i in range(N):
        X[i, dimensions[i]] = sides[i].float()

    return X


# Right-hand side of the Poisson equation
# TODO Use code from exp02 once it is merged
def f(X: Tensor) -> Tensor:
    """The right-hand side of the Poisson equation we aim to solve.

    Args:
        X: Batched quadrature points of shape (N, d_Omega).

    Returns:
        The function values as tensor of shape (N, 1).
    """
    d = X.shape[1:].numel()

    return d * pi**2 * prod(sin(pi * X), dim=1, keepdim=True)


# Manufactured solution
# TODO Use code from exp02 once it is merged
def u(X: Tensor) -> Tensor:
    """The solution of the Poisson equation we aim to solve.

    Args:
        X: Batched quadrature points of shape (N, d_Omega).

    Returns:
        The function values as tensor of shape (N, 1).
    """
    return prod(cos(pi * X), dim=1, keepdim=True)


def evaluate_interior_gramian(
    model: Module, X: Tensor, approximation: str
) -> Union[Tensor, List[Tensor]]:
    """Evaluate the interior loss' Gramian.

    Args:
        model: The model.
        X: Input for the interior loss.
        approximation: The approximation to use for the Gramian. Can be `'full'`,
            `'diagonal'`, or `'per_layer'`.

    Returns:
        The interior loss Gramian.

    Raises:
        ValueError: If the approximation is not one of `'full'`, `'diagonal'`, or
            `'per_layer'`.
    """
    batch_size = X.shape[0]
    param_names = [n for n, _ in model.named_parameters()]
    gramian = autograd_gramian(
        model, X, param_names, loss_type="interior", approximation=approximation
    )
    if approximation == "per_layer":
        return [g.div_(batch_size) for g in gramian]
    elif approximation in {"full", "diagonal"}:
        return gramian.div_(batch_size)
    else:
        raise ValueError(
            f"Unknown approximation {approximation!r}. "
            "Must be one of 'full', 'diagonal', or 'per_layer'."
        )


def evaluate_interior_loss(
    model: Union[Module, List[Module]], X: Tensor, y: Tensor
) -> Tuple[Tensor, Tensor, Union[List[Dict[str, Tensor]], None]]:
    """Evaluate the interior loss.

    Args:
        model: The model or a list of layers that form the sequential model. If the
            layers are supplied, the forward pass will use the more efficient forward
            Laplacian framework and return a list of dictionaries containing the push-
            forwards through all layers.
        X: Input for the interior loss.
        y: Target for the interior loss.

    Returns:
        The differentiable interior loss, differentiable residual, and intermediates
        of the computation graph that can be used to compute (approximate) curvature.

    Raises:
        ValueError: If the model is not a Module or a list of Modules.
    """
    # use autograd to compute the Laplacian
    if isinstance(model, Module):
        intermediates = None
        input_hessian = autograd_input_hessian(model, X)
        laplacian = einsum(input_hessian, "batch i i -> batch i")
    # use the forward Laplacian framework
    elif isinstance(model, list) and all(isinstance(layer, Module) for layer in model):
        intermediates = manual_forward_laplacian(model, X)
        laplacian = intermediates[-1]["laplacian"]
    else:
        raise ValueError(
            "Model must be a Module or a list of Modules that form a sequential model."
            f"Got: {model}."
        )
    residual = laplacian + y
    return 0.5 * ((residual) ** 2).mean(), residual, intermediates


def evaluate_boundary_gramian(
    model: Module, X, approximation: str
) -> Union[Tensor, List[Tensor]]:
    """Evaluate the boundary loss' Gramian.

    Args:
        model: The model.
        X: Input for the boundary loss.
        approximation: The approximation to use for the Gramian. Can be `'full'`,
            `'diagonal'`, or `'per_layer'`.

    Returns:
        The boundary loss Gramian.

    Raises:
        ValueError: If the approximation is not one of `'full'`, `'diagonal'`, or
            `'per_layer'`.
    """
    batch_size = X.shape[0]
    param_names = [n for n, _ in model.named_parameters()]
    gramian = autograd_gramian(
        model, X, param_names, loss_type="boundary", approximation=approximation
    )
    if approximation == "per_layer":
        return [g.div_(batch_size) for g in gramian]
    elif approximation in {"full", "diagonal"}:
        return gramian.div_(batch_size)
    else:
        raise ValueError(
            f"Unknown approximation {approximation!r}. "
            "Must be one of 'full', 'diagonal', or 'per_layer'."
        )


def evaluate_boundary_loss(
    model: Union[Module, List[Module]], X: Tensor, y: Tensor
) -> Tuple[Tensor, Tensor, Union[List[Tensor], None]]:
    """Evaluate the boundary loss.

    Args:
        model: The model.
        X: Input for the boundary loss.
        y: Target for the boundary loss.

    Returns:
        The differentiable boundary loss, the differentiable residual, and a list of
        intermediates of the computation graph that can be used to compute (approximate)
        curvature.

    Raises:
        ValueError: If the model is not a Module or a list of Modules.
    """
    if isinstance(model, Module):
        output = model(X)
        intermediates = None
    elif isinstance(model, list) and all(isinstance(layer, Module) for layer in model):
        intermediates = manual_forward(model, X)
        output = intermediates[-1]
    else:
        raise ValueError(
            "Model must be a Module or a list of Modules that form a sequential model."
            f"Got: {model}."
        )
    residual = output - y
    return 0.5 * ((residual) ** 2).mean(), residual, intermediates


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
    kfacs = check_layers_and_initialize_kfac(layers, initialize_to_identity=False)

    # We turn X's differentiability on to trigger the grad-output based
    # Kronecker factor computation by calling `grad` w.r.t. `X`. We restore the
    # original value later.
    X_original_requires_grad = X.requires_grad
    X.requires_grad = True

    # Compute the forward Laplacian and all the intermediates
    loss, residual, intermediates = evaluate_interior_loss(layers, X, y)

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

    # We used the residual in the loss and don't want its graph to be free
    # Therefore, set `retain_graph=True`.
    # trigger the backward hooks
    grad(residual, X, grad_outputs=ones_like(residual), retain_graph=True)

    # remove the hooks & reset original differentiability
    X.requires_grad = X_original_requires_grad
    for handle in handles:
        handle.remove()

    return loss, kfacs


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
    kfacs = check_layers_and_initialize_kfac(layers, initialize_to_identity=False)

    # We turn X's differentiability on to trigger the grad-output based
    # Kronecker factor computation by calling `grad` w.r.t. `X`. We restore the
    # original value later.
    X_original_requires_grad = X.requires_grad
    X.requires_grad = True

    # Compute the NN prediction, boundary loss, and all intermediates
    loss, residual, intermediates = evaluate_boundary_loss(layers, X, y)

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

    # We used the residual in the loss and don't want its graph to be freed.
    # Therefore, set `retain_graph=True`
    # trigger the backward hooks
    grad(residual, X, grad_outputs=ones_like(residual), retain_graph=True)

    # remove the hooks & reset original differentiability
    X.requires_grad = X_original_requires_grad
    for handle in handles:
        handle.remove()

    return loss, kfacs


def hook_add_output_based_kfac_expand(grad_t: Tensor, dest: Tensor) -> None:
    """Add the gradient's outer product into the destination tensor.

    Args:
        grad_t: The gradient w.r.t. the tensor `t` onto which this hook is installed.
        dest: The destination tensor onto which the gradient outer product is added
            in-place.
    """
    # flatten shared axes into one and use detached tensor for KFAC factor
    grad_t = rearrange(grad_t, "... d_in -> (...) d_in").detach()
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
