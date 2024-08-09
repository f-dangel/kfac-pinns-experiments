"""Functionality for solving the Poisson equation."""

from math import pi
from typing import Dict, List, Optional, Tuple, Union
from warnings import warn

from einops import einsum
from matplotlib import pyplot as plt
from torch import (
    Tensor,
    cat,
    cos,
    linspace,
    meshgrid,
    no_grad,
    ones,
    prod,
    rand,
    randint,
    sin,
    stack,
)
from torch import sum as torch_sum
from torch import zeros
from torch.autograd import grad
from torch.nn import Linear, Module
from tueplots import bundles

from kfac_pinns_exp.autodiff_utils import autograd_input_hessian
from kfac_pinns_exp.forward_laplacian import manual_forward_laplacian
from kfac_pinns_exp.kfac_utils import compute_kronecker_factors
from kfac_pinns_exp.pinn_utils import get_backpropagated_error
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
    if d == 10:
        norm = 1.34  # numerical approximation for dim_Omega=10
    elif d == 100:
        norm = 12.59  # numerical approximation for dim_Omega=100
    else:
        norm = 1.0
        warn(
            "[u_weinan_prods]: u_weinan_prods is not of unit L2 norm. "
            "Consider changing the normalization constant."
        )
    return X.reshape(N, d // 2, 2).prod(dim=2).sum(dim=1, keepdim=True) / norm


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
    if X.shape[-1] != 100:
        warn(
            "[u_weinan_norm]: u_weinan_norm is not of unit L2 norm. "
            "Consider changing the normalization constant."
        )
    norm = 33.47  # numerical approximation for dim_Omega=100
    return (X**2.0).sum(dim=1, keepdim=True) / norm


def f_weinan_norm(X: Tensor) -> Tensor:
    """The forcing corresponding to weinan_norm, identically 2 * dim_Omega.

    Args:
        X: Batched quadrature points of shape (N, d_Omega).

    Returns:
        2 * dim_Omega of the shape (len(X), 1).
    """
    N, d = X.shape
    return -2 * d * (1.0 / 33.47) * ones(N, 1)


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
    """
    loss, layer_inputs, layer_grad_outputs = (
        evaluate_interior_loss_with_layer_inputs_and_grad_outputs(
            layers, X, y, ggn_type
        )
    )
    kfacs = compute_kronecker_factors(
        layers, layer_inputs, layer_grad_outputs, ggn_type, kfac_approx
    )
    return loss, kfacs


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

            ax.plot(x, u_learned.cpu(), label="Normalized learned solution")
            ax.plot(x, u_true.cpu(), label="Normalized true solution", linestyle="--")
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
            ax[0].imshow(u_learned.cpu(), **kwargs)
            ax[1].imshow(u_true.cpu(), **kwargs)
            plt.savefig(savepath, bbox_inches="tight")
    else:
        raise ValueError(f"dim_Omega must be 1 or 2. Got {dim_Omega}.")

    plt.close(fig=fig)


def evaluate_interior_loss_with_layer_inputs_and_grad_outputs(
    layers: List[Module], X: Tensor, y: Tensor, ggn_type: str
) -> Tuple[Tensor, Dict[int, Tensor], Dict[int, Tensor]]:
    """Compute the interior loss, and inputs+output gradients of Linear layers.

    Args:
        layers: The list of layers that form the neural network.
        X: The input data.
        y: The target data.
        ggn_type: The type of GGN to use. Can be `'type-2'`, `'empirical'`, or
            `'forward-only'`.

    Returns:
        A tuple containing the loss, the inputs of the Linear layers, and the output
        gradients of the Linear layers. The layer inputs and output gradients are each
        combined into a matrix, and layer inputs are augmented with ones or zeros to
        account for the bias term.
    """
    layer_idxs = [
        idx
        for idx, layer in enumerate(layers)
        if (
            isinstance(layer, Linear)
            and layer.bias is not None
            and layer.bias.requires_grad
            and layer.weight.requires_grad
        )
    ]
    loss, residual, intermediates = evaluate_interior_loss(layers, X, y)

    layer_inputs = {}
    # layer inputs
    for idx in layer_idxs:
        # batch_size x d_in
        forward = intermediates[idx]["forward"]
        # batch_size x d_0 x d_in
        directional_gradients = intermediates[idx]["directional_gradients"]
        # batch_size x d_in
        laplacian = intermediates[idx]["laplacian"]
        # batch_size x (d_0 + 2) x (d_in + 1)
        layer_inputs[idx] = cat(  # noqa: B909
            [
                bias_augmentation(forward.detach(), 1).unsqueeze(1),
                bias_augmentation(directional_gradients.detach(), 0),
                bias_augmentation(laplacian.detach(), 0).unsqueeze(1),
            ],
            dim=1,
        )

    if ggn_type == "forward-only":
        return loss, layer_inputs, {}

    # compute all layer output gradients
    layer_outputs = sum(
        (
            [
                intermediates[idx + 1]["forward"],
                intermediates[idx + 1]["directional_gradients"],
                intermediates[idx + 1]["laplacian"],
            ]
            for idx in layer_idxs
        ),
        [],
    )
    # compute the gradient w.r.t. all relevant layer outputs
    error = get_backpropagated_error(residual, ggn_type)
    grad_outputs = list(
        grad(
            residual,
            layer_outputs,
            grad_outputs=error,
            # We used the residual in the loss and don't want its graph to be free
            # Therefore, set `retain_graph=True`.
            retain_graph=True,
            # only the Laplacian of the last layer output is used, hence the
            # directional gradients and forward outputs of the last layer are
            # not used. Hence we must set this flag to true and also enable
            # `materialize_grads` which sets these gradients to explicit zeros.
            allow_unused=True,
            materialize_grads=True,
        )
    )

    # collect all layer output gradients
    layer_grad_outputs = {}
    for idx in layer_idxs:
        # batch_size x d_out
        grad_forward = grad_outputs.pop(0)
        # batch_size x d_0 x d_out
        grad_directional_gradients = grad_outputs.pop(0)
        # batch_size x d_out
        grad_laplacian = grad_outputs.pop(0)
        # batch_size x (d_0 + 2) x d_out
        layer_grad_outputs[idx] = cat(  # noqa: B909
            [
                grad_forward.detach().unsqueeze(1),
                grad_directional_gradients.detach(),
                grad_laplacian.detach().unsqueeze(1),
            ],
            dim=1,
        )

    return loss, layer_inputs, layer_grad_outputs
