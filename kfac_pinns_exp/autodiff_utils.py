"""Utility functions for automatic differentiation."""

from typing import List, Tuple

from einops import einsum, rearrange
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


def autograd_gram_grads(
    model: Module, X: Tensor, param_names: List[str], detach: bool = True
) -> Tuple[Tensor]:
    """Compute the gradients used in the Gramian.

    Args:
        model: The model whose Laplacian's Gramian is considered. Must produce
            scalar outputs.
        X: The input to the model. First dimension is the batch dimension.
        param_names: List of unique parameter names forming the block.
        detach: Whether to detach the gradients from the computational graph.
            Default: `True`.

    Returns:
        The Gramian's gradients `gᵢ = ∇_θ {Tr[∇ₓ²f(xᵢ, θ)}` w.r.t. the specified parameters
        in tuple format. For each parameter `p`, the Gram gradient has shape
        `[batch_size, *p.shape]`.
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

    argnums = tuple(range(1, len(param_names) + 1))
    gram_grads = vmap(grad(laplacian, argnums=argnums))

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
    gram_grads = cat(
        [
            rearrange(g, "batch ... -> batch (...)")
            for g in autograd_gram_grads(model, X, param_names)
        ],
        dim=1,
    )
    return einsum(gram_grads, gram_grads, "batch i, batch j -> i j")
