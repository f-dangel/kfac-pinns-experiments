"""General utility functions."""

from typing import List

from torch import Tensor, cat


def separate_into_tiles(mat: Tensor, dims: List[int]) -> List[List[Tensor]]:
    """Separate a matrix into tiles of given dimensions.

    Args:
        mat: The matrix to be separated into tiles.
        dims: The tile dimensions.

    Returns:
        A list of lists of tiles.

    Example:
        >>> dims = [1, 2]
        >>> mat = Tensor([[1, 2, 5], [3, 4, 6], [7, 8, 9]]) # 3x3
        >>> separate_into_tiles(mat, dims) # [[1x1, 1x2], [1x2, 2x2]]
        [[tensor([[1.]]), tensor([[2., 5.]])], [tensor([[3.],
                [7.]]), tensor([[4., 6.],
                [8., 9.]])]]
    """
    row_tiles = mat.split(dims)
    return [list(row_tile.split(dims, dim=1)) for row_tile in row_tiles]


def combine_tiles(tiles: List[List[Tensor]]) -> Tensor:
    """Combine tiles into a single matrix.

    Args:
        tiles: The tiles to be combined.

    Returns:
        The combined matrix.

    Example:
        >>> tiles = [
        ...     [
        ...         Tensor([[1, 2], [3, 4]],), # 2x2
        ...         Tensor([[5], [6]]), # 2x1
        ...     ],
        ...     [
        ...         Tensor([[7, 8]]), # 1x2
        ...         Tensor([[9]]) # 1x1
        ...     ],
        ... ]
        >>> combine_tiles(tiles)
        tensor([[1., 2., 5.],
                [3., 4., 6.],
                [7., 8., 9.]])
    """
    row_tiles = [cat(col_tiles, dim=1) for col_tiles in tiles]
    return cat(row_tiles, dim=0)


def exponential_moving_average(dest: Tensor, update: Tensor, factor: float) -> None:
    """Update the destination tensor with an exponential moving average.

    `dest = factor * dest + (1 - factor) * update`

    Args:
        dest: The destination tensor that will be updated.
        update: The update tensor to be incorporated.
        factor: The exponential moving average factor. Must be in [0, 1).

    Raises:
        ValueError: If `factor` is not in [0, 1).
    """
    if not 0.0 <= factor < 1.0:
        raise ValueError(
            f"Exponential moving average factor must be in [0, 1). Got {factor}."
        )
    dest.mul_(factor).add_(update, alpha=1 - factor)
