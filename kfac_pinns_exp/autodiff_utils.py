"""Utility functions for automatic differentiation."""

from einops import einsum
from torch import Tensor
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


def autograd_gramian(model: Module, X: Tensor, param_name: str) -> Tensor:
    """Compute the model Laplacian's Gramian block stemming from a parameter.

    Args:
        model: The model whose Gramian will be computed. Must produce
            scalars as output.
        X: The input to the model. First dimension is the batch dimension.
        param_name: The name of the parameter whose Gramian block is computed.

    Returns:
        The Gramian block of the model Laplacian w.r.t. the parameter.
        If `θ` is the parameter, its Gramian has shape `[*θ.shape, *θ.shape]`:
        `∑ᵢ gᵢ @ gᵢᵀ` where `gᵢ = ∇_θ {Tr[∇ₓ²f(xᵢ, θ)}`.
    """
    # freeze all other parameters
    param_dict = {name: p for name, p in model.named_parameters() if name != param_name}

    def f(x: Tensor, param: Parameter) -> Tensor:
        """Forward pass on an un-batched input.

        Args:
            x: Un-batched 1d input.
            param: The parameter whose Gramian block is computed.

        Returns:
            Un-batched scalar output.
        """
        return functional_call(model, {**param_dict, param_name: param}, x)

    def laplacian(x: Tensor, param: Parameter) -> Tensor:
        """Compute the Laplacian of the model for an un-batched input.

        Args:
            x: Un-batched 1d input.
            param: The parameter whose Gramian block is computed.

        Returns:
            The scalar-valued Laplacian, i.e. Tr[∇ₓ²f(x, θ)].
        """
        hess_f = hessian(f, 0)  # (x, θ) → ∇ₓ²f(x, θ)
        return einsum(hess_f(x, param), "batch d d ->")

    def gramian(x: Tensor, param: Parameter) -> Tensor:
        """Compute the Gramian block of the model Laplacian for an un-batched input.

        Args:
            x: Un-batched 1d input.
            param: The parameter whose Gramian block is computed.

        Returns:
            The Gramian block of the model Laplacian, i.e. `g @ gᵀ` where
            `g = ∇_θ {Tr[∇ₓ²f(x, θ)}`. If `θ` is the parameter, the Gramian has shape
            `[*θ.shape, *θ.shape]`.
        """
        grad_laplacian = grad(laplacian, 1)  # (x, θ) → ∇_θ {Tr[∇ₓ²f(x, θ)]}
        d_laplacian_flat = grad_laplacian(x, param).detach().flatten()
        gramian_flat = einsum(d_laplacian_flat, d_laplacian_flat, "i,j->i j")
        return gramian_flat.reshape(*param.shape, *param.shape)

    param = model.get_parameter(param_name)
    return sum(gramian(x_n, param) for x_n in X)
