"""Implements functionality to solve the Fokker-Planck equation in log-space."""

from typing import Callable, Dict, List, Optional, Tuple, Union
from warnings import warn

from einops import einsum
from matplotlib import pyplot as plt
from torch import Tensor, linspace, meshgrid, no_grad, stack
from torch.nn import Module, Sequential
from tueplots import bundles

from kfac_pinns_exp import fokker_planck_equation
from kfac_pinns_exp.autodiff_utils import (
    autograd_input_divergence,
    autograd_input_hessian,
    autograd_input_jacobian,
)
from kfac_pinns_exp.log_fokker_planck_isotropic_equation import q_isotropic_gaussian
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
    mu_X = mu(X)
    div_mu = autograd_input_divergence(mu, X, coordinates=list(range(1, dim)))

    if isinstance(model, list) and all(isinstance(layer, Module) for layer in model):
        warn("Inefficient implementation")
        model = Sequential(*model)

    if isinstance(model, Module):
        intermediates = None

        # compute first derivatives of q
        jac_q = autograd_input_jacobian(model, X).reshape(batch_size, dim)
        dq_dt = jac_q[:, [0]]
        nabla_q = jac_q[:, 1:]

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
    sigma_T_nabla_q = einsum(sigma_X, nabla_q, "batch i k, batch i -> batch k")
    norm_sigma_T_nabla_q = (sigma_T_nabla_q**2).sum(dim=1, keepdim=True)
    nabla_q_mu = einsum(nabla_q, mu_X, "batch i, batch i -> batch").unsqueeze(-1)
    residual = (
        dq_dt
        + div_mu
        + nabla_q_mu
        - 0.5 * norm_sigma_T_nabla_q
        - 0.5 * tr_sigma_outer_hessian
    )
    loss = 0.5 * (residual**2).mean()
    return loss, residual, intermediates


evaluate_boundary_loss = fokker_planck_equation.evaluate_boundary_loss


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
            `'gaussian'`.
        dim_Omega: The dimension of the domain Omega. Can be `1` or `2`.
        model: The neural network model representing the learned solution.
        savepath: The path to save the plot.
        title: The title of the plot. Default: None.
        usetex: Whether to use LaTeX for rendering text. Default: `True`.

    Raises:
        ValueError: If `dim_Omega` is not `1` or `2`.
    """
    u = {"gaussian": q_isotropic_gaussian}[condition]
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
