"""Implements optimizers for PINNs.

Also implements argument parsing helpers for the optimizers and other built-in ones.
"""

from argparse import Namespace
from typing import List, Tuple

from torch.nn import Module
from torch.optim import SGD, Adam, Optimizer

from kfac_pinns_exp.optim.adam import parse_Adam_args
from kfac_pinns_exp.optim.engd import ENGD, parse_ENGD_args
from kfac_pinns_exp.optim.kfac import KFAC, parse_KFAC_args
from kfac_pinns_exp.optim.sgd import parse_SGD_args


def set_up_optimizer(
    layers: List[Module], optimizer: str, verbose: bool = False
) -> Tuple[Optimizer, Namespace]:
    """Parse arguments for the specified optimizer and construct it.

    Args:
        layers: The layers of the model.
        optimizer: The name of the optimizer to be used.
        verbose: Whether to print the parsed arguments. Default: `False`.

    Returns:
        The optimizer and the parsed arguments.
    """
    cls, parser_func = {
        "KFAC": (KFAC, parse_KFAC_args),
        "SGD": (SGD, parse_SGD_args),
        "Adam": (Adam, parse_Adam_args),
        "ENGD": (ENGD, parse_ENGD_args),
    }[optimizer]

    if optimizer in {"KFAC", "ENGD"}:
        param_representation = layers
    else:
        param_representation = sum((list(layer.parameters()) for layer in layers), [])

    prefix = f"{optimizer}_"

    args = parser_func(verbose=verbose, prefix=prefix)
    args_dict = vars(args)  # each key has a prefix that needs to be removed
    args_dict = {key.removeprefix(prefix): value for key, value in args_dict.items()}

    return cls(param_representation, **args_dict), args
