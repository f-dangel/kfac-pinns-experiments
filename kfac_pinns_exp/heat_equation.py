"""Functionality for solving the heat equation."""

from math import pi
from typing import Dict, List, Tuple, Union

from einops import einsum, rearrange, reduce
from torch import Tensor, cat, rand, zeros
from torch.autograd import grad
from torch.nn import Module

from kfac_pinns_exp.autodiff_utils import (
    autograd_input_hessian,
    autograd_input_jacobian,
)
from kfac_pinns_exp.forward_laplacian import manual_forward_laplacian
from kfac_pinns_exp.kfac_utils import check_layers_and_initialize_kfac
from kfac_pinns_exp.manual_differentiation import manual_forward
from kfac_pinns_exp.poisson_equation import get_backpropagated_error, square_boundary
from kfac_pinns_exp.utils import bias_augmentation


def square_boundary_random_time(N: int, dim: int) -> Tensor:
    """Draw points from the square boundary at random time.

    Args:
        N: The number of points to draw.
        dim: The dimension of the square.

    Returns:
        The points drawn from the square boundary at random time. Has shape
        `(N, 1 + dim)`. First entry along the second axis is time.
    """
    times = rand(N, 1)
    X_boundary = square_boundary(N, dim)
    return cat([times, X_boundary], dim=1)


def unit_square_at_start(N: int, dim: int) -> Tensor:
    """Draw points from the unit square at time 0.

    Args:
        N: The number of points to draw.
        dim: The dimension of the square.

    Returns:
        The points drawn from the unit square at time 0. Has shape
        `(N, 1 + dim)`. First entry along the second axis is time.
    """
    times = zeros(N, 1)
    X_square = rand(N, dim)
    return cat([times, X_square], dim=1)


def u_sin_product(X: Tensor) -> Tensor:
    """Solution of the heat equation with sine product initial conditions.

    (And zero boundary conditions.)

    Args:
        X: The points at which to evaluate the solution. First axis is batch dimension.
            Second axis is time, followed by spatial dimensions.

    Returns:
        The value of the solution at the given points. Has shape `(X.shape[0], 1)`.
    """
    dim_Omega = X.shape[-1] - 1
    time, spatial = X.split([1, dim_Omega], dim=-1)
    scale = -(pi**2 * dim_Omega) / 4
    return (scale * time).exp() * (pi * spatial).sin().prod(dim=-1, keepdim=True)


def evaluate_interior_loss(
    model: Union[Module, List[Module]], X: Tensor, y: Tensor
) -> Tuple[Tensor, Tensor, Union[List[Dict[str, Tensor]], None]]:
    """Evaluate the interior loss.

    Args:
        model: The model or a list of layers that form the sequential model. If the
            layers are supplied, the forward pass will use the forward Laplacian
            framework to compute the derivatives and return a list of dictionaries
            containing the push-forwards through all layers.
        X: Input for the interior loss.
        y: Target for the interior loss (all-zeros tensor).

    Returns:
        The differentiable interior loss, differentiable residual, and intermediates
        of the computation graph that can be used to compute (approximate) curvature.

    Raises:
        ValueError: If the model is not a Module or a list of Modules.
    """
    (_, d0) = X.shape
    spatial = list(range(1, d0))

    # use autograd to compute the Laplacian and time derivative
    if isinstance(model, Module):
        intermediates = None
        # slice away the time dimension of the Hessian
        spatial_hessian = autograd_input_hessian(model, X, coordinates=spatial)
        spatial_laplacian = einsum(spatial_hessian, "batch i i -> batch").unsqueeze(1)
        # slice away the spatial dimensions of the Jacobian
        time_jacobian = autograd_input_jacobian(model, X).squeeze(1)[:, [0]]
    # use forward Laplacian
    elif isinstance(model, list) and all(isinstance(layer, Module) for layer in model):
        intermediates = manual_forward_laplacian(model, X, coordinates=spatial)
        spatial_laplacian = intermediates[-1]["laplacian"]
        # slice away the spatial dimensions of the Jacobian
        time_jacobian = intermediates[-1]["directional_gradients"][:, 0]
    else:
        raise ValueError(
            "Model must be a Module or a list of Modules that form a sequential model."
            f"Got: {model}."
        )

    residual = time_jacobian - spatial_laplacian / 4 - y
    return 0.5 * (residual**2).mean(), residual, intermediates


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
    return 0.5 * (residual**2).mean(), residual, intermediates


def evaluate_interior_loss_and_kfac(
    layers: List[Module],
    X: Tensor,
    y: Tensor,
    ggn_type: str = "type-2",
    kfac_approx: str = "expand",
) -> Tuple[Tensor, Dict[int, Tuple[Tensor, Tensor]]]:
    """Evaluate the interior loss and compute its KFAC-expand approximation.

    Args:
        layers: The list of layers in the neural network.
        X: The input data.
        y: The target data.
        ggn_type: The type of GGN to compute. Can be `'empirical'`, `'type-2'`,
            or `'forward-only'`. Default: `'type-2'`.
        kfac_approx: The type of KFAC approximation to use. Can be `'expand'` or
            `'reduce'`. Default: `'expand'`.

    Returns:
        The (differentiable) interior loss and a dictionary whose keys are the layer
        indices and whose values are the two Kronecker factors.

    Raises:
        ValueError: If `kfac_approx` is not `'expand'` or `'reduce'`.
    """
    if kfac_approx not in {"expand", "reduce"}:
        raise ValueError(
            f"kfac_approx must be 'expand' or 'reduce'. Got {kfac_approx}."
        )
    kfacs = check_layers_and_initialize_kfac(layers, initialize_to_identity=False)

    # Compute the spatial Laplacian and time Jacobian and all the intermediates
    loss, residual, intermediates = evaluate_interior_loss(layers, X, y)

    ###############################################################################
    #                    COMPUTE INPUT-BASED KRONECKER FACTORS                    #
    ###############################################################################
    for layer_idx, (A, _) in kfacs.items():
        layer_in = intermediates[layer_idx]
        # batch_size x d_in
        forward = layer_in["forward"]
        # batch_size x d_0 x d_in
        directional_gradients = layer_in["directional_gradients"]
        # batch_size x d_in
        laplacian = layer_in["laplacian"]
        # batch_size x (d_0 + 2) x (d_in + 1)
        T = cat(
            [
                bias_augmentation(forward.detach(), 1).unsqueeze(1),
                bias_augmentation(directional_gradients.detach(), 0),
                bias_augmentation(laplacian.detach(), 0).unsqueeze(1),
            ],
            dim=1,
        )
        if kfac_approx == "expand":
            T = rearrange(T, "batch d_0 d_in -> (batch d_0) d_in")
        else:  # KFAC-reduce
            T = reduce(T, "batch d_0 d_in -> batch d_in", "mean")
        A.add_(T.T @ T, alpha=1 / T.shape[0])

    if ggn_type == "forward-only":
        # set all grad-output Kronecker factors to identity, no backward pass required
        for _, B in kfacs.values():
            B.fill_diagonal_(1.0)
        return loss, kfacs

    ###############################################################################
    #                   COMPUTE OUTPUT-BASED KRONECKER FACTORS                    #
    ###############################################################################
    error = get_backpropagated_error(residual, ggn_type)

    # compute the gradient w.r.t. all relevant layer outputs
    outputs = []
    for layer_idx in kfacs:
        layer_out = intermediates[layer_idx + 1]
        outputs.extend(
            [
                layer_out["forward"],
                layer_out["directional_gradients"],
                layer_out["laplacian"],
            ]
        )
    # We used the residual in the loss and don't want its graph to be free
    # Therefore, set `retain_graph=True`.
    grad_outputs = list(
        grad(
            residual,
            outputs,
            grad_outputs=error,
            retain_graph=True,
            # only the Laplacian of the last layer output is used, hence the
            # directional gradients and forward outputs of the last layer are
            # not used. Hence we must set this flag to true and also enable
            # `materialize_grads` which sets these gradients to explicit zeros.
            allow_unused=True,
            materialize_grads=True,
        )
    )

    # concatenate gradients w.r.t. all layer outputs into grad_T and form B
    batch_size = X.shape[0]
    for _, B in kfacs.values():
        # batch_size x d_out
        grad_forward = grad_outputs.pop(0)
        # batch_size x d_0 x d_out
        grad_directional_gradients = grad_outputs.pop(0)
        # batch_size x d_out
        grad_laplacian = grad_outputs.pop(0)
        # batch_size x (d_0 + 2) x d_out
        grad_T = cat(
            [
                grad_forward.detach().unsqueeze(1),
                grad_directional_gradients.detach(),
                grad_laplacian.detach().unsqueeze(1),
            ],
            dim=1,
        )
        if kfac_approx == "expand":
            grad_T = rearrange(grad_T, "batch d_0 d_out -> (batch d_0) d_out")
        else:  # KFAC-reduce
            grad_T = reduce(grad_T, "batch d_0 d_out -> batch d_out", "sum")
        B.add_(grad_T.T @ grad_T, alpha=batch_size)

    return loss, kfacs


def evaluate_boundary_loss_and_kfac(
    layers: List[Module],
    X: Tensor,
    y: Tensor,
    ggn_type: str = "type-2",
    kfac_approx: str = "expand",
) -> Tuple[Tensor, Dict[int, Tuple[Tensor, Tensor]]]:
    """Evaluate the boundary loss and compute its KFAC-expand approximation.

    Args:
        layers: The list of layers in the neural network.
        X: The input data.
        y: The target data.
        ggn_type: The type of GGN to compute. Can be `'empirical'`, `'type-2'`,
            or `'forward-only'`. Default: `'type-2'`.
        kfac_approx: The type of KFAC approximation to use. Can be `'expand'` or
            `'reduce'`. Default: `'expand'`.

    Returns:
        The (differentiable) boundary loss and a dictionary whose keys are the layer
        indices and whose values are the two Kronecker factors.

    Raises:
        ValueError: If `kfac_approx` is not `'expand'` or `'reduce'`.
    """
    if kfac_approx not in {"expand", "reduce"}:
        raise ValueError(
            f"kfac_approx must be 'expand' or 'reduce'. Got {kfac_approx}."
        )
    kfacs = check_layers_and_initialize_kfac(layers, initialize_to_identity=False)

    # Compute the NN prediction, boundary loss, and all intermediates
    loss, residual, intermediates = evaluate_boundary_loss(layers, X, y)

    ###############################################################################
    #                    COMPUTE INPUT-BASED KRONECKER FACTORS                    #
    ###############################################################################
    for layer_idx, (A, _) in kfacs.items():
        # no weight sharing here, hence KFAC expand and reduce are identical
        x = bias_augmentation(intermediates[layer_idx], 1).detach()
        A.add_(x.T @ x, alpha=1 / x.shape[0])

    if ggn_type == "forward-only":
        # set all grad-output Kronecker factors to identity, no backward pass required
        for _, B in kfacs.values():
            B.fill_diagonal_(1.0)
        return loss, kfacs

    ###############################################################################
    #                   COMPUTE OUTPUT-BASED KRONECKER FACTORS                    #
    ###############################################################################
    error = get_backpropagated_error(residual, ggn_type)

    # compute the gradient w.r.t. all relevant layer outputs
    outputs = [intermediates[layer_idx + 1] for layer_idx in kfacs]
    # We used the residual in the loss and don't want its graph to be free
    # Therefore, set `retain_graph=True`.
    grad_outputs = list(grad(residual, outputs, grad_outputs=error, retain_graph=True))

    # compute the grad-output covariance for each layer
    batch_size = X.shape[0]
    for _, B in kfacs.values():
        grad_out = grad_outputs.pop(0).detach()
        # no weight sharing here, hence KFAC expand and reduce are identical
        B.add_(grad_out.T @ grad_out, alpha=batch_size)

    return loss, kfacs
