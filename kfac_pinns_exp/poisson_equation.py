"""Functionality for solving the Poisson equation."""

from math import pi
from typing import Callable, Dict, List, Optional, Tuple, Union

from einops import einsum, rearrange, reduce
from matplotlib import pyplot as plt
from torch import (
    Tensor,
    cat,
    cos,
    linspace,
    meshgrid,
    no_grad,
    ones,
    ones_like,
    prod,
    rand,
    randint,
    sin,
    stack,
)
from torch import sum as torch_sum
from torch import zeros
from torch.autograd import grad
from torch.nn import Module
from tueplots import bundles

from kfac_pinns_exp.autodiff_utils import autograd_input_hessian
from kfac_pinns_exp.forward_laplacian import manual_forward_laplacian
from kfac_pinns_exp.kfac_utils import check_layers_and_initialize_kfac
from kfac_pinns_exp.manual_differentiation import manual_forward
from kfac_pinns_exp.utils import bias_augmentation


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


def f_sin_product(X: Tensor) -> Tensor:
    """The right-hand side of the Prod sine Poisson equation we aim to solve.

    Args:
        X: Batched quadrature points of shape (N, d_Omega).

    Returns:
        The function values as tensor of shape (N, 1).
    """
    d = X.shape[1:].numel()

    return d * pi**2 * prod(sin(pi * X), dim=1, keepdim=True)


def u_sin_product(X: Tensor) -> Tensor:
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


def u_weinan_prods(X: Tensor) -> Tensor:
    """A harmonic mixed polynomial of second order. Weinan uses dim=10.

    This example is taken from Weinans paper on the deep Ritz method:
    https://arxiv.org/abs/1710.00211
    It is a simple polynomial of degree two consisting of mixed products. It is
    thus harmonic, i.e., its Laplacian is zero.

    Args:
        X: Batched quadrature points of shape (N, d_Omega).

    Returns:
        The function values as tensor of shape (N, 1).
    """
    N, d = X.shape
    return X.reshape(N, d // 2, 2).prod(dim=2).sum(dim=1, keepdim=True)


def f_weinan_prods(X: Tensor) -> Tensor:
    """The forcing corresponding to weinan_prods, identically zero.

    Args:
        X: Batched quadrature points of shape (N, d_Omega).

    Returns:
        Zeros of the same shape (len(X), 1).
    """
    return zeros((len(X), 1))


def u_weinan_norm(X: Tensor) -> Tensor:
    """The squared norm. Weinan uses dim=100.

    This example is taken from Weinans paper on the deep Ritz method:
    https://arxiv.org/abs/1710.00211
    It is simply |x|^2 in 100d. The Laplacian is constant with value 200.

    Args:
        X: Batched quadrature points of shape (N, d_Omega).

    Returns:
        The function values as tensor of shape (N, 1).
    """
    return (X**2.0).sum(dim=1, keepdim=True)


def f_weinan_norm(X: Tensor) -> Tensor:
    """The forcing corresponding to weinan_norm, identically 2 * dim_Omega.

    Args:
        X: Batched quadrature points of shape (N, d_Omega).

    Returns:
        2 * dim_Omega of the shape (len(X), 1).
    """
    N, d = X.shape
    return -2 * d * ones(N, 1)


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
    """Evaluate the interior loss and compute its KFAC approximation.

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
    """Evaluate the boundary loss and compute its KFAC approximation.

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


@no_grad()
def plot_solution(
    condition: str,
    dim_Omega: int,
    model: Module,
    savepath: str,
    title: Optional[str] = None,
    usetex: bool = False,
):
    """Visualize the learned and true solution of the Poisson equation.

    Args:
        condition: String describing the boundary conditions of the PDE. Can be either
            `'sin_product'` or `'cos_sum'`.
        dim_Omega: The dimension of the domain Omega. Can be either `1` or `2`.
        model: The neural network model representing the learned solution.
        savepath: The path to save the plot.
        title: The title of the plot. Default: None.
        usetex: Whether to use LaTeX for rendering text. Default: `True`.

    Raises:
        ValueError: If `dim_Omega` is not `1` or `2`.
    """
    u = {"sin_product": u_sin_product, "cos_sum": u_cos_sum}[condition]
    ((dev, dt),) = {(p.device, p.dtype) for p in model.parameters()}

    if dim_Omega == 1:
        # set up grid, evaluate learned and true solution
        x = linspace(0, 1, 50).to(dev, dt).unsqueeze(1)
        u_learned = model(x).squeeze(1)
        u_true = u(x).squeeze(1)
        x.squeeze_(1)

        # normalize to [0; 1]
        u_true = (u_true - u_true.min()) / (u_true.max() - u_true.min())
        u_learned = (u_learned - u_learned.min()) / (u_learned.max() - u_learned.min())

        # plot
        with plt.rc_context(bundles.neurips2023(rel_width=1.0, ncols=1, usetex=usetex)):
            fig, ax = plt.subplots(1, 1)
            ax.set_xlabel("$x$")
            ax.set_ylabel("$u(x)$")
            if title is not None:
                ax.set_title(title)

            ax.plot(x, u_learned, label="Normalized learned solution")
            ax.plot(x, u_true, label="Normalized true solution", linestyle="--")
            ax.legend()
            plt.savefig(savepath, bbox_inches="tight")

    elif dim_Omega == 2:
        # set up grid, evaluate learned and true solution
        x, y = linspace(0, 1, 50).to(dev, dt), linspace(0, 1, 50).to(dev, dt)
        x_grid, y_grid = meshgrid(x, y, indexing="ij")
        xy_flat = stack([x_grid.flatten(), y_grid.flatten()], dim=1)
        u_learned = model(xy_flat).reshape(x_grid.shape)
        u_true = u(xy_flat).reshape(x_grid.shape)

        # normalize to [0; 1]
        u_learned = (u_learned - u_learned.min()) / (u_learned.max() - u_learned.min())
        u_true = (u_true - u_true.min()) / (u_true.max() - u_true.min())

        # plot
        with plt.rc_context(bundles.neurips2023(rel_width=1.0, ncols=1, usetex=usetex)):
            fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)
            ax[0].set_title("Normalized learned solution")
            ax[1].set_title("Normalized true solution")
            ax[0].set_xlabel("$x_1$")
            ax[1].set_xlabel("$x_1$")
            ax[0].set_ylabel("$x_2$")
            if title is not None:
                fig.suptitle(title, y=0.975)

            kwargs = {
                "vmin": 0,
                "vmax": 1,
                "interpolation": "none",
                "extent": [0, 1, 0, 1],
                "origin": "lower",
            }
            ax[0].imshow(u_learned, **kwargs)
            ax[1].imshow(u_true, **kwargs)
            plt.savefig(savepath, bbox_inches="tight")
    else:
        raise ValueError(f"dim_Omega must be 1 or 2. Got {dim_Omega}.")

    plt.close(fig=fig)
