"""General utility functions."""

from subprocess import CalledProcessError, CompletedProcess, run
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


def bias_augmentation(t: Tensor, augmentation: int, dim: int = -1) -> Tensor:
    """Augment a tensor to account for the bias contribution.

    Args:
        t: The tensor to be augmented.
        augmentation: The augmentation type. 0 for zeros, 1 for ones.
        dim: The dimension to augment. Default is the last dimension.

    Returns:
        The augmented tensor whose `dim` dimension is increased by 1.
    """
    dim = dim if dim > 0 else dim + t.ndim
    augmentation_fn = {0: t.new_zeros, 1: t.new_ones}[augmentation]
    augmentation_shape = t.shape[:dim] + (1,) + t.shape[dim + 1 :]
    return cat([t, augmentation_fn(augmentation_shape)], dim=dim)


def latex_float(number: float) -> str:
    """Convert a floating point number to a LaTeX formatted string.

    Args:
        number: The number to convert.

    Returns:
        str: The LaTeX formatted string.
    """
    float_str = f"{number:.1e}"
    base, exponent = float_str.split("e")
    return r"{0} \cdot 10^{{{1}}}".format(base, int(exponent))


def run_verbose(cmd: List[str]) -> CompletedProcess:
    """Run a command and print stdout & stderr if it fails.

    Args:
        cmd: The command to run.

    Returns:
        CompletedProcess: The result of the command.

    Raises:
        CalledProcessError: If the command fails.
    """
    try:
        job = run(cmd, capture_output=True, text=True, check=True)
        print("STDOUT:", job.stdout)
        print("STDERR:", job.stderr)
        return job
    except CalledProcessError as e:
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise e
