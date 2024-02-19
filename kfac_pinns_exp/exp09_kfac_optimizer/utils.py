"""Utility functions."""

from argparse import ArgumentParser, Namespace
from sys import argv


def parse_known_args_and_remove_from_argv(parser: ArgumentParser) -> Namespace:
    """Parse known arguments and remove them from `sys.argv`.

    See https://stackoverflow.com/a/35733750.

    Args:
        parser: An `ArgumentParser` object.

    Returns:
        A namespace with the parsed arguments.
    """
    args, left = parser.parse_known_args()
    argv[1:] = left
    return args


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

    args = parse_known_args_and_remove_from_argv(parser)

    if verbose:
        print(f"SGD arguments: {args}")

    return args


def parse_Adam_args(verbose: bool = False) -> Namespace:
    """Parse command-line arguments for the Adam optimizer.

    Args:
        verbose: Whether to print the parsed arguments. Default: `False`.

    Returns:
        A namespace with the parsed arguments.
    """
    parser = ArgumentParser(description="Adam optimizer parameters")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for the Adam optimizer.",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="Exponential decay rate for the first moment estimates of Adam.",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="Exponential decay rate for the second moment estimates of Adam.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Term added to Adam's denominator to improve numerical stability.",
    )
    args = parse_known_args_and_remove_from_argv(parser)

    # replace beta1 and beta2 with a tuple betas
    args.betas = (args.beta1, args.beta2)
    delattr(args, "beta1")
    delattr(args, "beta2")

    if verbose:
        print(f"Adam arguments: {args}")

    return args
