"""Implementation of the forward Laplacian framework from Li et. al 2023."""

from typing import Dict, List, Optional, Union

from einops import einsum
from torch import Tensor, eye, gt, stack, zeros_like
from torch.nn import Linear, Module, ReLU, Sigmoid, Tanh


def manual_forward_laplacian(  # noqa: C901
    layers: List[Module],
    x: Tensor,
    coordinates: Optional[List[int]] = None,
    coefficients: Optional[Union[Tensor, List[Tensor]]] = None,
) -> List[Dict[str, Tensor]]:
    """Compute the NN prediction and Laplacian (or weighted sum of second derivatives).

    Args:
        layers: A list of layers defining the NN.
        x: The input to the first layer. First dimension is batch dimension.
        coordinates: List of indices specifying the Hessian diagonal entries
            that are summed into the Laplacian. If `None`, all diagonal entries
            are summed. Default: `None`.
        coefficients: A coefficient matrix that, if specified, will compute a weighted
            sum `∑ᵢⱼ cᵢⱼ ∂²f/∂xᵢ∂xⱼ` instead of the Laplacian. If `None`, the Laplacian
            is computed, i.e. `cᵢⱼ = δᵢⱼ`. If `coordinates` is specified, the
            coefficients must match the size of the sub-space implied by `coordinates`.
            The coefficients can be specified in different formats:
            - Directly as square matrix `A` such that `cᵢⱼ = A[i, j]`.
            - In outer product form, as list of vectors `[v₁, ..., v_N]` such that
              `cᵢⱼ = [∑_n v_n v_nᵀ ]ᵢⱼ`. This structure is useful to reduce computation.

    Returns:
        A list of dictionaries, each containing the Taylor coefficients (0th-, 1st-, and
        summed 2nd-order) pushed through the layers layers.

    Raises:
        ValueError: If a layer uses in-place operations.
        ValueError: If coordinates are not unique or out of range.
        ValueError: If the coefficients shape does not match the expected shape.
    """
    # inialize Taylor coefficients
    batch_size, feature_dims = x.shape[0], x.shape[1:]
    num_features = feature_dims.numel()
    directional_grad_init = (
        eye(num_features, dtype=x.dtype, device=x.device)
        .unsqueeze(0)
        .repeat(batch_size, 1, 1)
    ).reshape(batch_size, num_features, *feature_dims)
    laplacian_init = zeros_like(x)
    derivatives = {
        "forward": x,
        "directional_gradients": directional_grad_init,
        "laplacian": laplacian_init,
    }

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

    if coefficients is not None:
        dim = num_features if coordinates is None else len(coordinates)
        if isinstance(coefficients, Tensor):
            expected_shape = (dim, dim)
            if coefficients.shape != expected_shape:
                raise ValueError(
                    f"Expected coefficients of shape {expected_shape}."
                    f" Got {coefficients.shape}."
                )
        elif isinstance(coefficients, list) and all(
            isinstance(c, Tensor) for c in coefficients
        ):
            expected_shape = (dim,)
            if any(c.shape != expected_shape for c in coefficients):
                raise ValueError(
                    f"Expected coefficients of shape {expected_shape}."
                    f" Got {[c.shape for c in coefficients]}."
                )
        else:
            raise ValueError(
                f"Expected coefficients to be a tensor or a list of tensors."
                f" Got {type(coefficients)}."
            )

    # pass Taylor coefficients through the network
    result = [derivatives]

    for layer in layers:
        if getattr(layer, "inplace", False):
            raise ValueError(
                f"Layers with in-place operations are not supported. Got {layer}."
            )
        derivatives = manual_forward_laplacian_layer(
            layer, derivatives, coordinates=coordinates, coefficients=coefficients
        )
        result.append(derivatives)

    return result


def manual_forward_laplacian_layer(
    layer: Module,
    derivatives: Dict[str, Tensor],
    coordinates: Union[List[int], None],
    coefficients: Union[Tensor, List[Tensor], None],
) -> Dict[str, Tensor]:
    """Propagate the 0th, 1st, and summed 2nd-order Taylor coefficients through a layer.

    Args:
        layer: The layer to propagate the Taylor coefficients through.
        derivatives: A dictionary containing the Taylor coefficients.
        coordinates: List of indices specifying the Hessian diagonal entries
            that are summed into the Laplacian. If `None`, all diagonal entries
            are summed.
        coefficients: A coefficient matrix that, if specified, will compute a weighted
            sum `∑ᵢⱼ cᵢⱼ ∂²f/∂xᵢ∂xⱼ` instead of the Laplacian. If `None`, the Laplacian
            is computed, i.e. `cᵢⱼ = δᵢⱼ`. If `coordinates` is specified, the
            coefficients must match the size of the sub-space implied by `coordinates`.
            The coefficients can be specified in different formats:
            - Directly as square matrix `A` such that `cᵢⱼ = A[i, j]`.
            - In outer product form, as list of vectors `[v₁, ..., v_N]` such that
              `cᵢⱼ = [∑_n v_n v_nᵀ ]ᵢⱼ`. This structure is useful to reduce computation.

    Returns:
        A dictionary containing the new Taylor coefficients.

    Raises:
        NotImplementedError: If the layer type is not supported.
        ValueError: If the coefficients are specified in incorrect format.
    """
    old_forward = derivatives["forward"]
    old_directional_gradients = derivatives["directional_gradients"]
    old_laplacian = derivatives["laplacian"]

    new_forward = layer(old_forward)

    if isinstance(layer, Linear):  # linear layers
        new_directional_gradients = einsum(
            layer.weight,
            old_directional_gradients,
            "d_out d_in, n d0 ... d_in -> n d0 ... d_out",
        )
        new_laplacian = einsum(
            layer.weight, old_laplacian, "d_out d_in, ... d_in -> ... d_out"
        )
    elif isinstance(layer, (Sigmoid, ReLU, Tanh)):  # element-wise activations
        if isinstance(layer, Sigmoid):
            jac = new_forward * (1 - new_forward)
            hess = jac * (1 - 2 * new_forward)
        elif isinstance(layer, ReLU):
            jac = gt(old_forward, 0).to(old_forward.dtype)
            hess = None
        elif isinstance(layer, Tanh):
            jac = 1 - new_forward**2
            hess = -2 * new_forward * jac

        new_directional_gradients = einsum(
            old_directional_gradients, jac, "n d0 ..., n ... -> n d0 ..."
        )

        # first-order contribution to Laplacian
        new_laplacian = einsum(jac, old_laplacian, "n ..., n ... -> n ... ")

        # second-order contribution to Laplacian (zero for ReLU)
        if hess is not None:
            # only use the relevant coordinates for the Laplacian
            coordinate_gradients = (
                old_directional_gradients
                if coordinates is None
                else old_directional_gradients[:, coordinates]
            )
            if coefficients is None or isinstance(coefficients, list):
                if isinstance(coefficients, list):
                    coefficients = stack(coefficients)
                    coordinate_gradients = einsum(
                        coefficients, coordinate_gradients, "c d0, n d0 ... -> n c ..."
                    )
                contribution = einsum(
                    hess, coordinate_gradients**2, "n ..., n d0 ... -> n ..."
                )
            elif isinstance(coefficients, Tensor):
                contribution = einsum(
                    hess,
                    coordinate_gradients,
                    coordinate_gradients,
                    coefficients,
                    "n ..., n i ..., n j ..., i j -> n ...",
                )
            else:
                raise ValueError(f"Unsupported coefficients: {coefficients}")

            new_laplacian.add_(contribution)
    else:
        raise NotImplementedError(f"Layer type not supported: {layer}.")

    return {
        "forward": new_forward,
        "directional_gradients": new_directional_gradients,
        "laplacian": new_laplacian,
    }
