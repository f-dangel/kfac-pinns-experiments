"""Implements argument parser for SGD."""

from argparse import ArgumentParser, Namespace

from kfac_pinns_exp.parse_utils import parse_known_args_and_remove_from_argv


def parse_Adam_args(verbose: bool = False, prefix: str = "Adam_") -> Namespace:
    """Parse command-line arguments for the Adam optimizer.

    Args:
        verbose: Whether to print the parsed arguments. Default: `False`.
        prefix: Prefix for the arguments. Default: "Adam_".

    Returns:
        A namespace with the parsed arguments.
    """
    parser = ArgumentParser(description="Adam optimizer parameters")
    parser.add_argument(
        f"--{prefix}lr",
        type=float,
        default=0.001,
        help="Learning rate for the Adam optimizer.",
    )
    parser.add_argument(
        f"--{prefix}beta1",
        type=float,
        default=0.9,
        help="Exponential decay rate for the first moment estimates of Adam.",
    )
    parser.add_argument(
        f"--{prefix}beta2",
        type=float,
        default=0.999,
        help="Exponential decay rate for the second moment estimates of Adam.",
    )
    parser.add_argument(
        f"--{prefix}eps",
        type=float,
        default=1e-8,
        help="Term added to Adam's denominator to improve numerical stability.",
    )
    args = parse_known_args_and_remove_from_argv(parser)

    # replace beta1 and beta2 with a tuple betas
    beta1, beta2 = f"{prefix}beta1", f"{prefix}beta2"
    beta = f"{prefix}betas"
    setattr(args, beta, (getattr(args, beta1), getattr(args, beta2)))
    delattr(args, beta1)
    delattr(args, beta2)

    if verbose:
        print(f"Adam arguments: {args}")

    return args
