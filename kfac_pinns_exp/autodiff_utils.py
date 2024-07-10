"""Utility functions for automatic differentiation."""

from typing import Callable, List, Optional, Union

from torch import Tensor
from torch.func import hessian, jacrev, vmap
from torch.nn import Module


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
