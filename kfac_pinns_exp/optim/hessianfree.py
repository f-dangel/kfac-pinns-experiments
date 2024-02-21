"""Argument parser and integration with the Hessian-free optimizer.

The implementation is at https://github.com/ltatzel/PyTorchHessianFree.
"""

from argparse import ArgumentParser, Namespace

from kfac_pinns_exp.parse_utils import parse_known_args_and_remove_from_argv


def parse_HessianFree_args(
    verbose: bool = True, prefix: str = "HessianFree_"
) -> Namespace:
    """Parse command-line arguments for the Hessian-free optimizer.

    Args:
        verbose: Whether to print the parsed arguments. Default: `True`.
        prefix: Prefix for the arguments. Default: "HessianFree_".

    Returns:
        A namespace with the parsed arguments.
    """
    parser = ArgumentParser(description="Hessian-free optimizer parameters.")
    parser.add_argument(
        f"--{prefix}curvature_opt",
        type=str,
        default="ggn",
        choices=["ggn", "hessian"],
        help="Curvature matrix used for the local quadratic approximation of the loss.",
    )
    parser.add_argument(
        f"--{prefix}damping",
        type=float,
        default=1.0,
        help="Tikhonov damping parameter.",
    )
    parser.add_argument(
        f"--{prefix}no_adapt_damping",
        dest=f"{prefix}adapt_damping",
        default=True,
        action="store_false",
        help="Whether to deactivate adaptive damping and use constant damping.",
    )
    parser.add_argument(
        f"--{prefix}cg_max_iter",
        type=int,
        default=250,
        help="Maximum number of CG iterations.",
    )
    parser.add_argument(
        f"--{prefix}cg_decay_x0",
        type=float,
        default=0.95,
        help="Decay factor of the previous CG solution used as init for the next.",
    )
    parser.add_argument(
        f"--{prefix}no_use_cg_backtracking",
        dest=f"{prefix}use_cg_backtracking",
        default=True,
        action="store_false",
        help="Whether to disable CG backtracking.",
    )
    parser.add_argument(
        f"--{prefix}lr",
        type=float,
        default=1.0,
        help="Learning rate.",
    )
    parser.add_argument(
        f"--{prefix}no_use_linesearch",
        dest=f"{prefix}use_linesearch",
        default=True,
        action="store_false",
        help="Whether to disable line search",
    )
    parser.add_argument(
        f"--{prefix}verbose",
        dest=f"{prefix}verbose",
        default=False,
        action="store_true",
        help="Whether to print internals to the command line.",
    )
    args = parse_known_args_and_remove_from_argv(parser)

    if verbose:
        print(f"Hessian-free arguments: {args}")

    return args
