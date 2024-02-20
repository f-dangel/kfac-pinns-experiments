"""Argparse utility functions."""

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


def check_all_args_parsed():
    """Make sure all command line arguments were parsed.

    Raises:
        ValueError: If there are unparsed arguments.
    """
    if len(argv) != 1:
        raise ValueError(f"The following arguments could not be parsed: {argv[1:]}.")
