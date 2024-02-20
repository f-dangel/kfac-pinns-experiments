"""Implements argument parser for LBFGS."""

from argparse import ArgumentParser, Namespace

from kfac_pinns_exp.parse_utils import parse_known_args_and_remove_from_argv


def parse_LBFGS_args(verbose: bool = False, prefix: str = "LBFGS_") -> Namespace:
    """Parse arguments of the LBFGS optimizer.

    Args:
        verbose: Whether to print the parsed arguments. Default: `False`.
        prefix: Prefix for the arguments. Default: `'LBFGS_'`.

    Returns:
        A namespace with the parsed arguments.
    """
    parser = ArgumentParser(description="LBFGS optimizer parameters.")

    parser.add_argument(
        f"--{prefix}lr",
        type=float,
        default=1.0,
        help="Learning rate.",
    )
    parser.add_argument(
        f"--{prefix}max_iter",
        type=int,
        default=20,
        help="Maximum number of iterations per optimization step.",
    )
    parser.add_argument(
        f"--{prefix}history_size",
        type=int,
        default=100,
        help="Update history size.",
    )
    args = parse_known_args_and_remove_from_argv(parser)

    if verbose:
        print(f"Adam arguments: {args}")

    return args
