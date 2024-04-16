"""Functionality for solving the Poisson equation."""

from math import pi
from typing import Callable, Dict, List, Tuple, Union

from einops import einsum, rearrange, reduce
from torch import Tensor, cat, cos, ones_like, prod, rand, randint, sin
from torch import sum as torch_sum
from torch.autograd import grad
from torch.nn import Module

from kfac_pinns_exp.autodiff_utils import autograd_gramian, autograd_input_hessian
from kfac_pinns_exp.forward_laplacian import manual_forward_laplacian
from kfac_pinns_exp.kfac_utils import check_layers_and_initialize_kfac
from kfac_pinns_exp.manual_differentiation import manual_forward
from kfac_pinns_exp.utils import bias_augmentation


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
    """The right-hand side of the Prod sine Poisson equation we aim to solve.

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
    """Prod sine solution of the Poisson equation we aim to solve.

    Args:
        X: Batched quadrature points of shape (N, d_Omega).

    Returns:
        The function values as tensor of shape (N, 1).
    """
    return prod(sin(pi * X), dim=1, keepdim=True)


def u_cos_sum(X: Tensor) -> Tensor:
    """Sum cosine solution of the Poisson equation we aim to solve.

    Args:
        X: Batched quadrature points of shape (N, d_Omega).

    Returns:
        The function values as tensor of shape (N, 1).
    """
    return torch_sum(cos(pi * X), dim=1, keepdim=True)


def f_cos_sum(X: Tensor) -> Tensor:
    """Sum cosine solution of the Poisson equation we aim to solve.

    Args:
        X: Batched quadrature points of shape (N, d_Omega).

    Returns:
        The function values as tensor of shape (N, 1).
    """
    return (pi**2) * torch_sum(cos(pi * X), dim=1, keepdim=True)


def l2_error(model: Module, X: Tensor, u: Callable[[Tensor], Tensor]) -> Tensor:
    """Computes the L2 norm of the error = model - u on the domain Omega.

    Args:
        model: The model.
        X: randomly drawn points in Omega.
        u: Function to evaluate the manufactured solution.

    Returns:
        The L2 norm of the error.
    """
    y = (model(X) - u(X)) ** 2
    return y.mean() ** (1 / 2)


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
        laplacian = einsum(input_hessian, "batch i i -> batch").unsqueeze(-1)
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
    return 0.5 * (residual**2).mean(), residual, intermediates


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

    # Compute the forward Laplacian and all the intermediates
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


def get_backpropagated_error(residual: Tensor, ggn_type: str) -> Tensor:
    """Get the error which is backpropagated to compute the second KFAC factor.

    Args:
        residual: The residual tensor which is squared then averaged to compute
            the loss.
        ggn_type: The type of GGN approximation. Can be "type-2" or "empirical".

    Returns:
        The error tensor. Has same shape as `residual`.

    Raises:
        NotImplementedError: If the `ggn_type` is not supported.
    """
    batch_size = residual.shape[0]
    if ggn_type == "type-2":
        return ones_like(residual) / batch_size
    elif ggn_type == "empirical":
        return residual.clone().detach() / batch_size
    raise NotImplementedError(f"GGN type {ggn_type} is not supported.")
