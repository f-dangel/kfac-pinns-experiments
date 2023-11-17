"""Utility functions for automatic differentiation."""

from torch import Tensor
from torch.func import hessian, vmap
from torch.nn import Module


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
