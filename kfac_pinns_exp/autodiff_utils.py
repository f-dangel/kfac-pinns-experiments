"""Utility functions for automatic differentiation."""

from typing import Callable, List, Optional, Tuple, Union

from einops import einsum, rearrange
from torch import Tensor, cat, ones
from torch.func import functional_call, grad, hessian, jacrev, vmap
from torch.nn import Module, Parameter

from kfac_pinns_exp import (
    fokker_planck_isotropic_equation,
    log_fokker_planck_isotropic_equation,
)


def autograd_input_divergence(
    model: Union[Module, Callable[[Tensor], Tensor]],
    X: Tensor,
    coordinates: Optional[List[int]] = None,
) -> Tensor:
    """Compute the divergence of the model w.r.t. its input.

    Args:
        model: The model whose divergence will be computed. Can either be an `nn.Module`
            or a tensor-to-tensor function.
        X: The input to the model. First dimension is the batch dimension.
            Must be differentiable.
        coordinates: List of indices specifying the coordinates w.r.t. which the
            divergence is taken. For example, if the function's arguments are 3d, but
            its output is 2d, we can specify `coordinates=[0, 1]` to compute the
            divergence w.r.t. the first two dimensions. Length of `coordinates` must
            correspond to the output dimension of the model. If `None`, the full
            divergence is computed. Default: `None`.

    Returns:
        The divergence of the model w.r.t. X. Has shape `[batch_size, 1]`.

    Raises:
        ValueError: If `coordinates` are specified but not unique or out of range.
    """
    num_features = X.shape[1:].numel()

    if coordinates is not None:
        if len(set(coordinates)) != len(coordinates) or len(coordinates) == 0:
            raise ValueError(
                f"Coordinates must be unique and non-empty. Got {coordinates}."
            )
        if any(c < 0 or c >= num_features for c in coordinates):
            raise ValueError(
                f"Coordinates must be in the range [0, {num_features})."
                f" Got {coordinates}."
            )

    def f(x: Tensor) -> Tensor:
        """Forward pass on an un-batched input.

        Args:
            x: Un-batched input.

        Returns:
            Un-batched output.

        Raises:
            ValueError: If the output shape does not match the combination of input
                shape and specified coordinates.
        """
        out = model(x)
        expected_shape = x.shape if coordinates is None else (len(coordinates),)
        if expected_shape != (
            out.shape if coordinates is None else (out.shape.numel(),)
        ):
            raise ValueError(
                "Output shape must match input shape or length of coordinates."
                f" Got {out.shape} output, {x.shape} input, {coordinates} coordinates."
            )
        return out

    def divergence(x: Tensor) -> Tensor:
        """Compute the divergence of the model w.r.t. its input.

        Args:
            x: Un-batched input.

        Returns:
            Un-batched divergence.
        """
        jac = jacrev(f)(x)
        if coordinates is None:
            jac = jac.reshape(x.numel(), x.numel())
        else:
            jac = jac.reshape(-1, x.numel())[:, coordinates]
        return jac.trace().unsqueeze(0)

    return vmap(divergence)(X)


def autograd_input_jacobian(
    model: Union[Module, Callable[[Tensor], Tensor]], X: Tensor
) -> Tensor:
    """Compute the batched Jacobian of the model w.r.t. its input.

    Args:
        model: The model whose Jacobian will be computed. Can either be an `nn.Module`
            or a tensor-to-tensor function.
        X: The input to the model. First dimension is the batch dimension.
            Must be differentiable.

    Returns:
        The Jacobian of the model w.r.t. X. Has shape
        `[batch_size, *model(X).shape[1:], *X.shape[1:]]`.
    """

    def f(x: Tensor) -> Tensor:
        """Forward pass on an un-batched input.

        Args:
            x: Un-batched input.

        Returns:
            Un-batched output.
        """
        return model(x)

    jac_f_X = vmap(jacrev(f))
    return jac_f_X(X)


def autograd_input_hessian(
    model: Union[Module, Callable[[Tensor], Tensor]],
    X: Tensor,
    coordinates: Optional[List[int]] = None,
) -> Tensor:
    """Compute the batched Hessian of the model w.r.t. its input.

    Args:
        model: The model whose Hessian will be computed. Must produce batched scalars as
            output. Can either be an `nn.Module` or a tensor-to-tensor function.
        X: The input to the model. First dimension is the batch dimension.
            Must be differentiable.
        coordinates: List of indices specifying the Hessian rows and columns to keep.
            If None, the full Hessian is computed. Default: `None`. Currently this
            feature only works if `X` is a batched vector.

    Returns:
        The Hessians of the model w.r.t. X. Has shape
        `[batch_size, *X.shape[1:], *X.shape[1:]]`.

    Raises:
        ValueError: If `coordinates` are specified but not unique or out of range.
        NotImplementedError: If `coordinates` is specified but the input is not a
            batched vector.
    """

    def f(x: Tensor) -> Tensor:
        """Forward pass on an un-batched input.

        Args:
            x: Un-batched 1d input.

        Returns:
            Un-batched scalar output.

        Raises:
            ValueError: If the input or output have incorrect shapes.
        """
        if x.ndim != 1:
            raise ValueError(f"Input must be 1d. Got {x.ndim}d.")

        output = model(x).squeeze()

        if output.ndim != 0:
            raise ValueError(f"Output must be 0d. Got {output.ndim}d.")

        return output

    hess_f_X = vmap(hessian(f))
    hess = hess_f_X(X)

    # slice rows and columns if coordinates are specified
    if coordinates is not None:
        if len(set(coordinates)) != len(coordinates) or len(coordinates) == 0:
            raise ValueError(
                f"Coordinates must be unique and non-empty. Got {coordinates}."
            )
        if X.ndim != 2:
            raise NotImplementedError(
                f"Coordinates only support batched vector (2d) inputs. Got {X.ndim}d."
            )
        _, num_features = X.shape
        if any(c < 0 or c >= num_features for c in coordinates):
            raise ValueError(
                f"Coordinates must be in the range [0, {num_features})."
                f" Got {coordinates}."
            )
        hess = hess[:, coordinates][:, :, coordinates]

    return hess


def autograd_gram_grads(
    model: Module,
    X: Tensor,
    param_names: List[str],
    detach: bool = True,
    loss_type: str = "poisson_interior",
) -> Tuple[Tensor]:
    """Compute the gradients used in the Gramian.

    Args:
        model: The model whose Laplacian's Gramian is considered. Must produce
            scalar outputs.
        X: The input to the model. First dimension is the batch dimension.
        param_names: List of unique parameter names forming the block.
        detach: Whether to detach the gradients from the computational graph.
            Default: `True`.
        loss_type: The type of loss. Either `'poisson_interior'`, `'poisson_boundary'`,
            `'heat_interior'`, `'heat_boundary'`, `'fokker-planck-isotropic_interior'`,
            `'fokker-planck-isotropic_boundary'`,
            `'log-fokker-planck-isotropic_interior'`, or.
            `'log-fokker-planck-isotropic_boundary'`, or.
            Default: `'poisson_interior'`.
            For `'poisson_interior'`, the Laplacian's gradients are computed.
            For `'heat_interior'`, the gradients of the difference between the
            temporal Jacobian and scaled spatial Laplacian are computed.
            For `'poisson_boundary'` and `'heat_boundary'`, the model's gradients are
            computed. Default: `'poisson_interior'`.

    Returns:
        The Gramian's gradients w.r.t. the specified parameters in tuple format:
        - `gᵢ = ∇_θ {Tr[∇ₓ²f(xᵢ, θ)]}` if `loss_type='poisson_interior'`
        - `gᵢ = ∇_θ {∇_t f((tᵢ, xᵢ), θ) - Tr[∇ₓ²f((tᵢ, xᵢ), θ)] / 4}` if
          `loss_type='heat_interior'`
        - `gᵢ = ∇_θ {f(xᵢ, θ)}` if `loss_type='poisson_boundary'`
        - `gᵢ = ∇_θ {f((tᵢ, xᵢ), θ)}` if `loss_type='heat_boundary'`
        For each parameter `p`, the Gram gradient has shape `[batch_size, *p.shape]`.
    """
    frozen = {
        name: p for name, p in model.named_parameters() if name not in param_names
    }

    def f(x: Tensor, *params: Parameter) -> Tensor:
        """Forward pass on an un-batched input.

        Args:
            x: Un-batched 1d input.
            params: The parameters forming the block of the Gramian in same order as
                supplied in `param_names`.

        Returns:
            Un-batched scalar output.
        """
        variable = dict(zip(param_names, params))
        return functional_call(model, frozen | variable, x).squeeze()

    def poisson_pde_operator(x: Tensor, *params: Parameter) -> Tensor:
        """Evaluate the Poisson equation's differential operator on an un-batched input.

        Args:
            x: Un-batched 1d input.
            params: The parameters forming the block of the Gramian in same order as
                supplied in `param_names`.

        Returns:
            The scalar-valued Laplacian, i.e. `Tr[∇ₓ²f(x, θ)]`.
        """
        hess_f = hessian(f, argnums=0)  # (x, θ) → ∇²ₓf(x, θ)
        return einsum(hess_f(x, *params), "d d ->")

    def heat_pde_operator(x: Tensor, *params: Parameter) -> Tensor:
        """Evaluate the heat equation's differential operator on an un-batched input.

        Args:
            x: Un-batched 1d input.
            params: The parameters forming the block of the Gramian in same order as
                supplied in `param_names`.

        Returns:
            The difference of time-Jacobian and spatial-Laplacian, i.e.
            `∇_t f((t, x), θ) - Tr[∇ₓ²f((t, x), θ)] / 4`.
        """
        hess_f = hessian(f, argnums=0)  # (x, θ) → ∇²_{(t, x)} f((t, x), θ)
        jacobian_f = jacrev(f, argnums=0)  # (x, θ) → ∇_{(t,x)} f((t, x), θ)

        # evaluate Hessian, remove temporal dimension and take Laplacian
        hess = hess_f(x, *params)[1:][:, 1:]
        laplacian = einsum(hess, "d d ->")

        # evaluate Jacobian, remove spatial dimensions
        jacobian = jacobian_f(x, *params)[0]

        return jacobian - laplacian / 4

    def fokker_planck_isotropic_pde_operator(x: Tensor, *params: Parameter) -> Tensor:
        """Evaluate the isotropic FP equation's differential operator.

        Args:
            x: Un-batched 1d input.
            params: The parameters forming the block of the Gramian in same order as
                supplied in `param_names`.

        Returns:
            The isotropic FP operator, i.e.
            `∂_t f(t, x) + divₓ(f(t, x) * μ(t, x)) - 0.5 * Tr( σσᵀ ∇²ₓ f(t, x) )
            = div( f(t, x) * (1, μ(t, x)) ) - 0.5 * Tr( σσᵀ ∇²ₓ f(t, x) )`.
        """

        def p_times_mu(x: Tensor, *params: Parameter) -> Tensor:
            """Compute the product between the probability density and the vector field.

            Args:
                x: Un-batched 1d input.
                params: The parameters forming the block of the Gramian in same order as
                    supplied in `param_names`.

            Returns:
                The product between the probability density and the vector field.
                Has the same shape as `x`.
            """
            p = f(x, *params)
            mu = fokker_planck_isotropic_equation.mu_isotropic(x)
            augment = ones(1, dtype=p.dtype, device=p.device)
            return p * cat([augment, mu])

        jacobian_p_times_mu = jacrev(p_times_mu, argnums=0)
        dp_dt_plus_div_p_times_mu = jacobian_p_times_mu(x, *params).trace()

        hess_f = hessian(f, argnums=0)  # (x, θ) → ∇²_{(t, x)} f((t, x), θ)
        sigma = fokker_planck_isotropic_equation.sigma_isotropic(
            x.unsqueeze(0)
        ).squeeze(0)
        hess = hess_f(x, *params)[1:][:, 1:]
        tr_sigma_outer_hess = einsum(sigma, sigma, hess, "i j, k j, k i->")

        return dp_dt_plus_div_p_times_mu - 0.5 * tr_sigma_outer_hess

    def log_fokker_planck_isotropic_pde_operator(
        x: Tensor, *params: Parameter
    ) -> Tensor:
        """Evaluate the isotropic log-FP equation's differential operator.

        Args:
            x: Un-batched 1d input.
            params: The parameters forming the block of the Gramian in same order as
                supplied in `param_names`.

        Returns:
            The isotropic FP operator, i.e.
            `∂_t f(t, x) + divₓ(μ(t, x)) + (∇ₓ f(t, x))ᵀ μ(t, x)
            - 0.5 * || σ(t, x)ᵀ ∇ₓ f(t, x) ||² - 0.5 * Tr( σσᵀ ∇²ₓ f(t, x) )`.
        """
        mu = log_fokker_planck_isotropic_equation.mu_isotropic
        sigma = log_fokker_planck_isotropic_equation.sigma_isotropic

        mu_x = mu(x)
        sigma_x = sigma(x.unsqueeze(0)).squeeze(0)
        jacobian_q = jacrev(f, argnums=0)(x, *params)
        dq_dt, dq_dx = jacobian_q[0], jacobian_q[1:]

        # compute divₓ(μ(t, x))
        div_mu_x = jacrev(mu)(x)[1:][:, 1:].trace()
        # compute (∇ₓ f(t, x))ᵀ μ(t, x)
        dq_dx_mu = einsum(dq_dx, mu_x, "i, i ->")
        # compute || σ(t, x)ᵀ ∇ₓ f(t, x) ||²
        norm_sigma_dq_dx = (einsum(sigma_x, dq_dx, "i j, i -> j") ** 2).sum()

        # compute Tr( σσᵀ ∇²ₓ q(t, x) )
        hess_f = hessian(f, argnums=0)  # (x, θ) → ∇²_{(t, x)} f((t, x), θ)
        sigma = fokker_planck_isotropic_equation.sigma_isotropic(
            x.unsqueeze(0)
        ).squeeze(0)
        hess = hess_f(x, *params)[1:][:, 1:]
        tr_sigma_outer_hess = einsum(sigma, sigma, hess, "i j, k j, k i->")

        return (
            dq_dt
            + div_mu_x
            + dq_dx_mu
            - 0.5 * norm_sigma_dq_dx
            - 0.5 * tr_sigma_outer_hess
        )

    # function that will be differentiated w.r.t. the parameters
    func = {
        "poisson_interior": poisson_pde_operator,
        "heat_interior": heat_pde_operator,
        "fokker-planck-isotropic_interior": fokker_planck_isotropic_pde_operator,
        "log-fokker-planck-isotropic_interior": log_fokker_planck_isotropic_pde_operator,
        "poisson_boundary": f,
        "heat_boundary": f,
        "fokker-planck-isotropic_boundary": f,
        "log-fokker-planck-isotropic_boundary": f,
    }[loss_type]
    argnums = tuple(range(1, len(param_names) + 1))

    gram_grads = vmap(grad(func, argnums=argnums))

    # need to replicate the parameters `batch_size` times
    batch_size = X.shape[0]
    params = []
    for name in param_names:
        p = model.get_parameter(name)
        keep = p.ndim * [-1]
        params.append(p.unsqueeze(0).expand(batch_size, *keep))

    result = gram_grads(X, *params)
    if detach:
        result = tuple(r.detach() for r in result)

    return result


def autograd_gramian(
    model: Module,
    X: Tensor,
    param_names: List[str],
    loss_type: str = "poisson_interior",
    approximation: str = "full",
) -> Union[Tensor, List[Tensor]]:
    """Compute a block of the model's (or its Laplacian's) Gramian.

    Args:
        model: The model whose Gramian will be computed. Must produce
            scalars as output.
        X: The input to the model. First dimension is the batch dimension.
        param_names: List of unique parameter names forming the block.
        loss_type: The type of loss. Either `'poisson_interior'`,
            `'poisson_boundary'`, `'heat_interior'`, or `'heat_boundary'`.
            For `'poisson_interior'`, the Laplacian's Gramian is computed.
            For `'poisson_boundary'`, and `'heat_boundary`, the model's Gramian is
            computed. For `'heat_interior'`, the Gramian of the difference of
            time-Jacobian and spatial-Laplacian is computed.
            Default: `'poisson_interior'`.
        approximation: The approximation to use for the Gramian. Either `'full'`,
            `'diagonal'`, or `'per_layer'`.

    Returns:
        The Gramian block of the model (or its Laplacian) w.r.t. the flattened and
        concatenated parameters. If `θ` is the flattened and concatenated parameter,
        its Gramian has shape `[*θ.shape, *θ.shape]`: `∑ᵢ gᵢ @ gᵢᵀ` where
        - `gᵢ = ∇_θ {Tr[∇ₓ²f(xᵢ, θ)]}` for `loss_type='poisson_interior'`
        - `gᵢ = ∇_θ f(xᵢ, θ)` for `loss_type='poisson_boundary'`
        - `gᵢ = ∇_θ {∇_t f((tᵢ, xᵢ), θ) - Tr[∇ₓ²f((tᵢ, xᵢ), θ)] / 4}`
          for `loss_type='heat_interior'`
        - `gᵢ = ∇_θ f((tᵢ, xᵢ), θ)` for `loss_type='heat_boundary'`

        If `approximation='diagonal'`, only the diagonal of shape `[*θ.shape]` is
        returned. If `approximation='per_layer'`, a list of Gramians is returned,
        one for each layer of the model.

    Raises:
        NotImplementedError: If the approximation is not implemented.
        ValueError: If parameters of the same layer are not contiguous in
            `param_names`.
    """
    gram_grads = cat(
        [
            rearrange(g, "batch ... -> batch (...)")
            for g in autograd_gram_grads(model, X, param_names, loss_type=loss_type)
        ],
        dim=1,
    )
    if approximation == "full":
        return einsum(gram_grads, gram_grads, "batch i, batch j -> i j")
    elif approximation == "diagonal":
        return gram_grads.pow_(2).sum(0)
    elif approximation == "per_layer":
        # construct blocks in terms of parameter names
        blocks = []

        current_layer, _ = param_names[0].rsplit(".")
        blocks.append([param_names[0]])

        for param_name in param_names[1:]:
            this_layer, _ = param_name.rsplit(".")
            if this_layer == current_layer:
                blocks[-1].append(param_name)
            else:
                blocks.append([param_name])
                current_layer = this_layer

        if sum(blocks, []) != param_names:
            raise ValueError(
                f"Parameter names must be contiguous by layer. Got {param_names}."
            )

        block_sizes = []
        for block in blocks:
            block_sizes.append(sum(model.get_parameter(name).numel() for name in block))

        gramians = [
            einsum(gram_grads_block, gram_grads_block, "batch i, batch j -> i j")
            for gram_grads_block in gram_grads.split(block_sizes, dim=1)
        ]
        return gramians
    else:
        raise NotImplementedError(f"Approximation {approximation!r} not implemented.")
