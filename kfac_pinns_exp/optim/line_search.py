"""Implements line search algorithms."""

from argparse import ArgumentParser
from typing import Any, Callable, Dict, List, Tuple, Union
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

    f_0 = f()
    f_values = []

    for alpha in grid:
        for param, orig, step in zip(params, original, params_step):
            param.data = orig + alpha * step
        f_values.append(f())

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


def parse_backtracking_line_search_args(
    verbose: bool = False, prefix: str = "backtracking_line_search_"
) -> Dict[str, Any]:
    """Parse command-line arguments for the backtracking line search.

    Args:
        verbose: Whether to print the parsed arguments. Default: `False`.
        prefix: Prefix for the arguments. Default: `'backtracking_line_search_'`.

    Returns:
        A dictionary containing the parsed arguments for the line search.
    """
    parser = ArgumentParser(description="Backtracking line search parameters.")
    parser.add_argument(
        f"--{prefix}init_alpha",
        type=float,
        help="Initial update step proposal.",
        default=1.0,
    )
    parser.add_argument(
        f"--{prefix}beta",
        type=float,
        help="Step size reduction factor.",
        default=0.8,
    )
    parser.add_argument(
        f"--{prefix}max_steps",
        type=int,
        help="Maximum number of steps",
        default=20,
    )
    args = parse_known_args_and_remove_from_argv(parser)

    if verbose:
        print("Parsed arguments for backtracking_line_search: ", args)

    args_dict = vars(args)  # each key has a prefix that needs to be removed
    args_dict = {key.removeprefix(prefix): value for key, value in args_dict.items()}
    return args_dict


@no_grad()
def backtracking_line_search(
    f: Callable[[], Tensor],
    params: List[Union[Tensor, Parameter]],
    params_step: List[Tensor],
    init_alpha=1.0,
    beta=0.8,
    c=1e-2,
    max_steps=20,
) -> Tuple[float, float]:
    """Perform a backtracking line search to find a 'good' step size.

    Update the parameters in-place using the 'good' step size.

    Iteratively reduces the step until the Armijo condition is satisfied.

    Args:
        f: The function to minimize.
        params: The parameters of the function with gradients stored under `.grad`.
        params_step: The step direction.
        init_alpha: The initial step size. Default: `1.0`.
        beta: The factor by which to reduce the step size. Default: `0.8`.
        c: The slope for the Armijo condition. Default: `1e-2`.
        max_steps: The maximum number of iterations. Default: `20`.

    Returns:
        The chosen step size and its associated function value.
    """
    original = [param.data.clone() for param in params]

    # current function value and directional derivative
    f_0 = f()
    c_direc_deriv = c * sum(
        g.flatten().dot(s.flatten())
        for g, s in zip([p.grad for p in params], params_step)
    )
    if c_direc_deriv >= 0.0:
        simplefilter("always", UserWarning)
        warn("Direction is not a descent direction.")

    # scale `step` until `f` is significantly reduced
    for s in range(max_steps):
        alpha = init_alpha * beta**s

        for param, orig, step in zip(params, original, params_step):
            param.data = orig + alpha * step

        f_alpha = f()

        if f_alpha <= f_0 + alpha * c_direc_deriv:
            return alpha, f_alpha.item()

    # no suitable update found
    simplefilter("always", UserWarning)
    warn("Line search could not find a decreasing value.")

    for param, orig in zip(params, original):
        param.data = orig

    return 0.0, f_0.item()
