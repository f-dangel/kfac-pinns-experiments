"""Functionality for solving the heat equation."""

from math import pi
from typing import Dict, List, Optional, Tuple, Union

from einops import einsum
from matplotlib import pyplot as plt
from torch import Tensor, cat, linspace, meshgrid, no_grad, rand, stack, zeros
from torch.autograd import grad
from torch.nn import Linear, Module
from tueplots import bundles

from kfac_pinns_exp.autodiff_utils import (
    autograd_input_hessian,
    autograd_input_jacobian,
)
from kfac_pinns_exp.forward_laplacian import manual_forward_laplacian
from kfac_pinns_exp.kfac_utils import compute_kronecker_factors
from kfac_pinns_exp.pinn_utils import get_backpropagated_error
from kfac_pinns_exp.plot_utils import create_animation
from kfac_pinns_exp.poisson_equation import square_boundary
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
    scale = -(pi**2) * dim_Omega / 4
    return (scale * time).exp() * (pi * spatial).sin().prod(dim=-1, keepdim=True)


def u_sin_sum(X: Tensor) -> Tensor:
    """Solution of the heat equation with sine sum initial & bdry conditions.

    (and zero right-hand side)

    The solution is u(t,x) = exp(-t)sum_i=1^d sin (alpha*pi*x_i)
    which has a vanishing right-hand side if alpha = 2/pi and the Laplacian
    is multiplied by 0.25 (as we do in this script).

    Args:
        X: The points at which to evaluate the solution. First axis is batch dimension.
            Second axis is time, followed by spatial dimensions.

    Returns:
        The value of the solution at the given points. Has shape `(X.shape[0], 1)`.
    """
    dim_Omega = X.shape[-1] - 1
    time, spatial = X.split([1, dim_Omega], dim=-1)

    alpha = 2.0 / pi

    return (-time).exp() * (alpha * pi * spatial).sin().sum(dim=-1, keepdim=True)


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
    # Compute the spatial Laplacian and time Jacobian and all the intermediates
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
    """Visualize the learned and true solution of the heat equation.

    Args:
        condition: String describing the boundary conditions of the PDE. Can be either
            `'sin_product'` or `'cos_sum'`.
        dim_Omega: The dimension of the domain Omega. Can be `1` or `2`.
        model: The neural network model representing the learned solution.
        savepath: The path to save the plot.
        title: The title of the plot. Default: None.
        usetex: Whether to use LaTeX for rendering text. Default: `True`.

    Raises:
        ValueError: If `dim_Omega` is not `1` or `2`.
    """
    u = {
        "sin_product": u_sin_product,
        "sin_sum": u_sin_sum,
    }[condition]
    ((dev, dt),) = {(p.device, p.dtype) for p in model.parameters()}

    imshow_kwargs = {
        "vmin": 0,
        "vmax": 1,
        "interpolation": "none",
        "extent": [0, 1, 0, 1],
        "origin": "lower",
    }

    if dim_Omega == 1:
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
            ax[0].set_xlabel("$x$")
            ax[1].set_xlabel("$x$")
            ax[0].set_ylabel("$t$")
            if title is not None:
                fig.suptitle(title, y=0.975)
            ax[0].imshow(u_learned.cpu(), **imshow_kwargs)
            ax[1].imshow(u_true.cpu(), **imshow_kwargs)
            plt.savefig(savepath, bbox_inches="tight")

        plt.close(fig=fig)

    elif dim_Omega == 2:
        ts = linspace(0, 1, 30).to(dev, dt)
        xs, ys = linspace(0, 1, 50).to(dev, dt), linspace(0, 1, 50).to(dev, dt)
        t_grid, x_grid, y_grid = meshgrid(ts, xs, ys, indexing="ij")
        txy_flat = stack([t_grid.flatten(), x_grid.flatten(), y_grid.flatten()], dim=1)
        u_true = u(txy_flat).reshape(*ts.shape, *xs.shape, *ys.shape)
        u_learned = model(txy_flat).reshape(*ts.shape, *xs.shape, *ys.shape)

        # normalize to [0; 1]
        u_learned = (u_learned - u_learned.min()) / (u_learned.max() - u_learned.min())
        u_true = (u_true - u_true.min()) / (u_true.max() - u_true.min())

        frames = []
        for idx, t in enumerate(ts):
            framepath = savepath.replace(".pdf", f"_frame_{idx:03g}.pdf")
            frames.append(framepath)
            # plot frame
            with plt.rc_context(
                bundles.neurips2023(rel_width=1.0, ncols=1, usetex=usetex)
            ):
                fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)
                ax[0].set_title("Normalized learned solution")
                ax[1].set_title("Normalized true solution")
                ax[0].set_xlabel("$x$")
                ax[1].set_xlabel("$x$")
                ax[0].set_ylabel("$t$")
                if title is not None:
                    fig.suptitle(title + f" ($t = {t:.2f})$", y=0.975)
            ax[0].imshow(u_learned[idx].cpu(), **imshow_kwargs)
            ax[1].imshow(u_true[idx].cpu(), **imshow_kwargs)
            plt.savefig(framepath, bbox_inches="tight")
            plt.close(fig)

        create_animation(frames, savepath.replace(".pdf", ".gif"))

    else:
        raise ValueError(f"dim_Omega must be 1 or 2. Got {dim_Omega}.")


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
