"""Utility functions for automatic differentiation."""

from typing import List

from einops import einsum
from torch import Tensor, cat
from torch.func import functional_call, grad, hessian, vmap
from torch.nn import Module, Parameter


def autograd_input_hessian(model: Module, X: Tensor) -> Tensor:
    """Compute the batched Hessian of the model w.r.t. its input.

    Args:
        model: The model whose Hessian will be computed. Must produce batched scalars as
            output.
        X: The input to the model. First dimension is the batch dimension.
            Must be differentiable.

    Returns:
        The Hessians of the model w.r.t. X. Has shape
        `[batch_size, *X.shape[1:], *X.shape[1:]]`.
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
    return hess_f_X(X)


def autograd_gramian(model: Module, X: Tensor, param_names: List[str]) -> Tensor:
    """Compute a block of the model Laplacian's Gramian.

    Args:
        model: The model whose Gramian will be computed. Must produce
            scalars as output.
        X: The input to the model. First dimension is the batch dimension.
        param_names: List of unique parameter names forming the block.

    Returns:
        The Gramian block of the model Laplacian w.r.t. the flattened and concatenated
        parameters. If `θ` is the flattened and concatenated parameter, its Gramian has
        shape `[*θ.shape, *θ.shape]`: `∑ᵢ gᵢ @ gᵢᵀ` where `gᵢ = ∇_θ {Tr[∇ₓ²f(xᵢ, θ)}`.
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
        return functional_call(model, frozen | variable, x)

    def laplacian(x: Tensor, *params: Parameter) -> Tensor:
        """Compute the Laplacian of the model for an un-batched input.

        Args:
            x: Un-batched 1d input.
            params: The parameters forming the block of the Gramian in same order as
                supplied in `param_names`.

        Returns:
            The scalar-valued Laplacian, i.e. Tr[∇ₓ²f(x, θ)].
        """
        hess_f = hessian(f, argnums=0)  # (x, θ) → ∇ₓ²f(x, θ)
        return einsum(hess_f(x, *params), "batch d d ->")

    def gramian(x: Tensor, *params: Parameter) -> Tensor:
        """Compute the Gramian block of the model Laplacian for an un-batched input.

        Args:
            x: Un-batched 1d input.
            params: The parameters forming the block of the Gramian in same order as
                supplied in `param_names`.

        Returns:
            The Gramian block of the model Laplacian, i.e. `g @ gᵀ` where
            `g = ∇_θ {Tr[∇ₓ²f(x, θ)}`. If `θ` are the flattened and concatenated
            parameters, the Gramian has shape `[*θ.shape, *θ.shape]`.
        """
        argnums = tuple(range(1, len(params) + 1))

        # (x, θ) → ∇_θ {Tr[∇ₓ²f(x, θ)]}
        grad_laplacian = grad(laplacian, argnums=argnums)

        gram_grad = grad_laplacian(x, *params)
        gram_grad = cat([g.detach().flatten() for g in gram_grad])
        return einsum(gram_grad, gram_grad, "i,j -> i j")

    params = tuple(model.get_parameter(name) for name in param_names)
    return sum(gramian(x_n, *params) for x_n in X)
