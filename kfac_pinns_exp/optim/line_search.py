"""Implements line search algorithms."""

from argparse import ArgumentParser
from typing import Callable, List, Tuple, Union
from warnings import simplefilter, warn

from torch import Tensor, logspace, no_grad
from torch.nn import Parameter

from kfac_pinns_exp.parse_utils import parse_known_args_and_remove_from_argv


def parse_grid_line_search_args(
    verbose: bool = False, prefix: str = "grid_line_search_"
) -> List[float]:
    """Parse command-line arguments for the grid line search.

    Args:
        verbose: Whether to print the parsed arguments. Default: `False`.
        prefix: Prefix for the arguments. Default: `'grid_line_search_'`.

    Returns:
        The grid values.
    """
    parser = ArgumentParser(description="Line grid search parameters.")
    parser.add_argument(
        f"--{prefix}log2min",
        type=float,
        help="Log2 of the minimum step size to try.",
        default=-30,
    )
    parser.add_argument(
        f"--{prefix}log2max",
        type=float,
        help="Log2 of the maximum step size to try.",
        default=0,
    )
    parser.add_argument(
        f"--{prefix}num_steps",
        type=int,
        help="Resolution of the logarithmic grid between min and max.",
        default=31,
    )
    args = parse_known_args_and_remove_from_argv(parser)

    if verbose:
        print("Parsed arguments for grid_line_search: ", args)

    log2min = getattr(args, f"{prefix}log2min")
    log2max = getattr(args, f"{prefix}log2max")
    num_steps = getattr(args, f"{prefix}num_steps")

    return logspace(log2min, log2max, num_steps, base=2).tolist()


@no_grad()
def grid_line_search(
    f: Callable[[], Tensor],
    params: List[Union[Tensor, Parameter]],
    params_step: List[Tensor],
    grid: List[float],
) -> Tuple[float, float]:
    """Perform a grid search to find the best step size.

    Update the parameters using the step size that leads to the
    smallest loss.

    Args:
        f: The function to minimize.
        params: The parameters of the function.
        params_step: The step direction.
        grid: The grid of step sizes to try.

    Returns:
        The best step size and its associated function value.
    """
    original = [param.data.clone() for param in params]

    f_0 = f().item()
    f_values = []

    for alpha in grid:
        for param, orig, step in zip(params, original, params_step):
            param.data = orig + alpha * step
        f_values.append(f().item())

    f_best = min(f_values)
    argbest = f_values.index(f_best)

    if f_0 < f_best:
        simplefilter("always", UserWarning)
        warn("Line search could not find a decreasing value.")
        best = 0
    else:
        best = grid[argbest]

    # update the parameters
    for param, orig, step in zip(params, original, params_step):
        param.data = orig + best * step

    return best, f_best
