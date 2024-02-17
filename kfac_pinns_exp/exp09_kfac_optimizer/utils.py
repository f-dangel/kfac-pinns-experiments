"""Utility functions."""

from argparse import ArgumentParser, Namespace


def parse_SGD_args(verbose: bool = False) -> Namespace:
    """Parse command-line arguments for the SGD optimizer.

    Args:
        verbose: Whether to print the parsed arguments. Default: `False`.

    Returns:
        A namespace with the parsed arguments.
    """
    parser = ArgumentParser(description="SGD optimizer parameters")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for the SGD optimizer.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0,
        help="Momentum for the SGD optimizer.",
    )
    args, _ = parser.parse_known_args()

    if verbose:
        print(f"SGD arguments: {args}")

    return args
