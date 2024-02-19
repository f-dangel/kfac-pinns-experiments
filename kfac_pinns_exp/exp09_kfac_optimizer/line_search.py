"""Line search algorithms."""

from argparse import ArgumentParser
from typing import Callable, List, Tuple, Union
from warnings import warn

from torch import Tensor, logspace, no_grad
from torch.nn import Parameter


def parse_grid_line_search_args() -> List[float]:
    """Parse command-line arguments for the grid line search.

    Returns:
        The grid values.
    """
    parser = ArgumentParser(description="Line grid search parameters.")
    parser.add_argument(
        "--grid_line_search_log2min",
        type=float,
        help="Log2 of the minimum step size to try.",
        default=-30,
    )
    parser.add_argument(
        "--grid_line_search_log2max",
        type=float,
        help="Log2 of the maximum step size to try.",
        default=0,
    )
    parser.add_argument(
        "--grid_line_search_num_steps",
        type=float,
        help="Resolution of the logarithmic grid between min and max.",
        default=31,
    )
    args, _ = parser.parse_known_args()

    return logspace(
        args.grid_line_search_log2min,
        args.grid_line_search_log2max,
        args.grid_line_search_num_steps,
        base=2,
    ).tolist()


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
        warn("Line search could not find a decreasing value. Skipping step.")
        best = 0
    else:
        best = grid[argbest]

    # update the parameters
    for param, orig, step in zip(params, original, params_step):
        param.data = orig + best * step

    return best, f_best
