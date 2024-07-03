"""Implements functionality to support solving the Fokker-Planck equation."""

from math import sqrt
from typing import Callable, Dict, List, Optional, Tuple, Union

from einops import einsum
from matplotlib import pyplot as plt
from torch import (
    Tensor,
    cat,
    eye,
    linspace,
    meshgrid,
    no_grad,
    ones,
    stack,
    zeros,
    zeros_like,
)
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import Module
from tueplots import bundles

from kfac_pinns_exp.autodiff_utils import (
    autograd_input_divergence,
    autograd_input_hessian,
)
from kfac_pinns_exp.forward_laplacian import manual_forward_laplacian
from kfac_pinns_exp.manual_differentiation import manual_forward
from kfac_pinns_exp.plot_utils import create_animation


def evaluate_interior_loss(
    model: Union[Module, List[Module]],
    X: Tensor,
    y: Tensor,
    mu: Callable[[Tensor], Tensor],
    sigma: Callable[[Tensor], Tensor],
) -> Tuple[Tensor, Tensor, Union[List[Dict[str, Tensor]], None]]:
    """Evaluate the interior loss.

    Args:
        model: The model or a list of layers that form the sequential model. If the
            layers are supplied, the forward pass will use the more efficient forward
            Laplacian framework and return a list of dictionaries containing the push-
            forwards through all layers.
        X: Input for the interior loss. Has shape `(batch_size, 1 + dim_Omega)`. One
            datum `X[n]` has coordinates `(t, x_1, x_2 , ..., x_dim_Omega)`.
        y: Target for the interior loss. Has shape `(batch_size, 1)`.
        mu: Vector field. Maps an un-batched input `x` to a tensor `mu(x)` of shape
            `(dim_Omega,)`.
        sigma: Diffusivity matrix. Maps `X` to a tensor `sigma(X)` of shape
            `(batch_size, dim_Omega, k)` with arbitrary `k` (usually `k = dim_Omega`).

    Returns:
        The differentiable interior loss, differentiable residual, and intermediates
        of the computation graph that can be used to compute (approximate) curvature.

    Raises:
        ValueError: If the model is not a PyTorch `Module` or a list of layers.
        NotImplementedError: If the sigma matrix is not identical for each datum in the
            batch.
    """
    batch_size, dim = X.shape
    sigma_X = sigma(X)
    sigma_outer = einsum(sigma_X, sigma_X, "batch i j, batch k j -> batch i k")

    if isinstance(model, list) and all(isinstance(layer, Module) for layer in model):
        # TODO Make sure that sigma_X is identical along batch dimension
        if not sigma_outer.allclose(
            sigma_outer[0].unsqueeze(0).expand(batch_size, -1, -1)
        ):
            raise NotImplementedError(
                "Sigma must be identical for each datum in the batch."
            )

        sigma_outer = sigma_outer[0]
        intermediates = manual_forward_laplacian(
            model, X, coordinates=list(range(1, dim)), coefficients=sigma_outer
        )
        tr_sigma_outer_hessian = intermediates[-1]["laplacian"]

        # compute div(p μ) = div[ p(t, x) μ(t, x) ] (fixed t) using the product rule
        dp_dx = intermediates[-1]["directional_gradients"][:, 1:].squeeze(-1)
        div_mu = autograd_input_divergence(mu, X, coordinates=list(range(1, dim)))
        mu_X = mu(X)
        p = intermediates[-1]["forward"]

        div_p_times_mu = (
            einsum(dp_dx, mu_X, "batch i, batch i -> batch").unsqueeze(-1) + p * div_mu
        )

        # compute ∂p/∂t + div(p μ)
        dp_dt = intermediates[-1]["directional_gradients"][:, 0]
        dp_dt_plus_div_p_times_mu = dp_dt + div_p_times_mu

    elif isinstance(model, Module):
        intermediates = None

        # compute div(p μ)
        def p_times_mu(x: Tensor) -> Tensor:
            """Compute the product between p(t, x) and the augmented μ(t, x).

            Args:
                x: Un-batched input (time and spatial coordinates).

            Returns:
                Product of p(t, x) and augmented μ(t, x).
            """
            mu_x = mu(x)
            augment = ones(1, dtype=mu_x.dtype, device=mu_x.device)
            mu_x_augmented = cat([augment, mu_x])
            return model(x) * mu_x_augmented

        dp_dt_plus_div_p_times_mu = autograd_input_divergence(p_times_mu, X)

        # compute Tr(σ σᵀ ∂²p/∂x²)
        hessian_X = autograd_input_hessian(model, X)  # [batch_size, d + 1, d + 1]
        hessian_spatial = hessian_X[:, 1:, 1:]  # [batch_size, d, d]
        sigma_outer_hessian = einsum(
            sigma_outer, hessian_spatial, "batch i k, batch k j -> batch i j"
        )
        tr_sigma_outer_hessian = einsum(
            sigma_outer_hessian, "batch i i -> batch"
        ).unsqueeze(-1)

    else:
        raise ValueError(
            f"Model must be a PyTorch Module or a list of layers. Got {model}."
        )

    # compute residual and loss
    residual = (dp_dt_plus_div_p_times_mu - 0.5 * tr_sigma_outer_hessian) - y
    loss = 0.5 * (residual**2).mean()

    return loss, residual, intermediates


def evaluate_boundary_loss(
    model: Union[Module, List[Module]], X: Tensor, y: Tensor
) -> Tuple[Tensor, Tensor, Union[List[Tensor], None]]:
    """Evaluate the boundary loss.

    Args:
        model: The model.
        X: Input for the boundary loss. Has shape `(batch_size, 1 + dim_Omega)`.
        y: Target for the boundary loss. Has shape `(batch_size, 1)`.

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


def mu_isotropic_gaussian(x: Tensor) -> Tensor:
    """Vector field for isotropic Gaussian case.

    Args:
        x: Un-batched input of shape `(1 + dim_Omega)` containing time and spatial
            coordinates, or batched input of shape `(batch_size, 1 + dim_Omega)`.

    Returns:
        The vector field as tensor of shape `(dim_Omega)`, or `(batch_size, dim_Omega)`
        if `x` is batched.
    """
    dim = x.shape[-1]
    _, spatial = x.split([1, dim - 1], dim=-1)
    return -0.5 * spatial


def sigma_isotropic_gaussian(X: Tensor) -> Tensor:
    """Diffusivity matrix for isotropic Gaussian case.

    Args:
        X: Batched input of shape `(batch_size, 1 + dim_Omega)` containing time and
            spatial coordinates.

    Returns:
        The diffusivity matrix as tensor of shape `(batch_size, dim_Omega, dim_Omega)`.
    """
    (batch_size, dim) = X.shape
    dim_Omega = dim - 1
    return (
        sqrt(2) * eye(dim_Omega, dtype=X.dtype, device=X.device).unsqueeze(0)
    ).expand(batch_size, dim_Omega, dim_Omega)


def p_isotropic_gaussian(X: Tensor) -> Tensor:
    """Isotropic Gaussian solution to the Fokker-Planck equation.

    Args:
        X: Batched quadrature points of shape `(N, d_Omega + 1)`.

    Returns:
        The function values as tensor of shape `(N, 1)`.
    """
    exp_t = (-X[:, 0]).exp()
    covariance = exp_t + 2 * (1 - exp_t)  # [batch_size]

    output = zeros_like(covariance)  # [batch_size]

    batch_size, d = X.shape
    d -= 1

    # TODO Implement more efficiently
    # (using torch.distributions.independent.Independent)
    mean = zeros(d, device=X.device, dtype=X.dtype)
    identity = eye(d, device=X.device, dtype=X.dtype)

    for n in range(batch_size):
        dist = MultivariateNormal(mean, covariance[n] * identity)
        spatial_n = X[n, 1:]
        output[n] = dist.log_prob(spatial_n).exp()

    return output.unsqueeze(-1)


@no_grad()
def plot_solution(
    condition: str,
    dim_Omega: int,
    model: Module,
    savepath: str,
    title: Optional[str] = None,
    usetex: bool = False,
):
    """Visualize the learned and true solution of the Fokker-Planck equation.

    Args:
        condition: String describing the boundary conditions of the PDE. Can be
            `'isotropic_gaussian'`.
        dim_Omega: The dimension of the domain Omega. Can be `1` or `2`.
        model: The neural network model representing the learned solution.
        savepath: The path to save the plot.
        title: The title of the plot. Default: None.
        usetex: Whether to use LaTeX for rendering text. Default: `True`.

    Raises:
        ValueError: If `dim_Omega` is not `1` or `2`.
    """
    u = {"isotropic_gaussian": p_isotropic_gaussian}[condition]
    ((dev, dt),) = {(p.device, p.dtype) for p in model.parameters()}

    imshow_kwargs = {
        "vmin": 0,
        "vmax": 1,
        "interpolation": "none",
        "extent": {1: [-5, 5, 0, 1], 2: [-5, 5, -5, 5]}[dim_Omega],
        "origin": "lower",
        "aspect": {1: 10, 2: None}[dim_Omega],
    }

    if dim_Omega == 1:
        # set up grid, evaluate learned and true solution
        x, y = linspace(0, 1, 50).to(dev, dt), linspace(-5, 5, 50).to(dev, dt)
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
            ax[0].imshow(u_learned, **imshow_kwargs)
            ax[1].imshow(u_true, **imshow_kwargs)
            plt.savefig(savepath, bbox_inches="tight")

        plt.close(fig=fig)

    elif dim_Omega == 2:
        ts = linspace(0, 1, 30).to(dev, dt)
        xs, ys = linspace(-5, 5, 50).to(dev, dt), linspace(-5, 5, 50).to(dev, dt)
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
                ax[0].set_ylabel("$y$")
                if title is not None:
                    fig.suptitle(title + f" ($t = {t:.2f})$", y=0.975)

            ax[0].imshow(u_learned[idx], **imshow_kwargs)
            ax[1].imshow(u_true[idx], **imshow_kwargs)
            plt.savefig(framepath, bbox_inches="tight")
            plt.close(fig)

        create_animation(frames, savepath.replace(".pdf", ".gif"))

    else:
        raise ValueError(f"dim_Omega must be 1 or 2. Got {dim_Omega}.")
