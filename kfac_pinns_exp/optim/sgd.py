"""Implements argument parser for SGD."""

from argparse import ArgumentParser, Namespace

from kfac_pinns_exp.parse_utils import parse_known_args_and_remove_from_argv


def parse_SGD_args(verbose: bool = False, prefix: str = "SGD_") -> Namespace:
    """Parse command-line arguments for the SGD optimizer.

    Args:
        verbose: Whether to print the parsed arguments. Default: `False`.
        prefix: Prefix for the arguments. Default: `'SGD_'`.

    Returns:
        A namespace with the parsed arguments.
    """
    parser = ArgumentParser(description="SGD optimizer parameters")
    parser.add_argument(
        f"--{prefix}lr",
        type=float,
        default=0.001,
        help="Learning rate for the SGD optimizer.",
    )
    parser.add_argument(
        f"--{prefix}momentum",
        type=float,
        default=0,
        help="Momentum for the SGD optimizer.",
    )

    args = parse_known_args_and_remove_from_argv(parser)

    if verbose:
        print(f"SGD arguments: {args}")

    return args
