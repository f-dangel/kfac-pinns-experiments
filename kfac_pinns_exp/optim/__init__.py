"""Implements optimizers for PINNs.

Also implements argument parsing helpers for the optimizers and other built-in ones.
"""

from argparse import Namespace
from typing import List, Tuple

from hessianfree.optimizer import HessianFree
from torch.nn import Module, Sequential
from torch.optim import LBFGS, SGD, Adam, Optimizer

from kfac_pinns_exp.optim.adam import parse_Adam_args
from kfac_pinns_exp.optim.engd import ENGD, parse_ENGD_args
from kfac_pinns_exp.optim.hessianfree import parse_HessianFree_args
from kfac_pinns_exp.optim.hessianfree_cached import (
    HessianFreeCached,
    parse_HessianFreeCached_args,
)
from kfac_pinns_exp.optim.kfac import KFAC, parse_KFAC_args
from kfac_pinns_exp.optim.lbfgs import parse_LBFGS_args
from kfac_pinns_exp.optim.sgd import parse_SGD_args


def set_up_optimizer(
    layers: List[Module], optimizer: str, equation: str, verbose: bool = False
) -> Tuple[Optimizer, Namespace]:
    """Parse arguments for the specified optimizer and construct it.

    Args:
        layers: The layers of the model.
        optimizer: The name of the optimizer to be used.
        equation: The name of the equation to be solved.
        verbose: Whether to print the parsed arguments. Default: `False`.

    Returns:
        The optimizer and the parsed arguments.
    """
    cls, parser_func = {
        "KFAC": (KFAC, parse_KFAC_args),
        "SGD": (SGD, parse_SGD_args),
        "Adam": (Adam, parse_Adam_args),
        "ENGD": (ENGD, parse_ENGD_args),
        "LBFGS": (LBFGS, parse_LBFGS_args),
        "HessianFree": (HessianFree, parse_HessianFree_args),
        "HessianFreeCached": (HessianFreeCached, parse_HessianFreeCached_args),
    }[optimizer]

    prefix = f"{optimizer}_"
    args = parser_func(verbose=verbose, prefix=prefix)
    args_dict = vars(args)  # each key has a prefix that needs to be removed
    args_dict = {key.removeprefix(prefix): value for key, value in args_dict.items()}

    # Some optimizers require passing the equation as argument. We parse this as general
    # argument and overwrite the entry from the optimizer's parser.
    if optimizer in {"KFAC", "ENGD", "HessianFreeCached"}:
        if verbose:
            print(
                f"Overwriting {optimizer}_equation={args_dict['equation']!r}"
                f" -> {equation!r} from general args."
            )
        args_dict["equation"] = equation

    if optimizer == "KFAC":
        param_representation = layers
    elif optimizer == "ENGD":
        param_representation = Sequential(*layers)
    elif optimizer == "HessianFreeCached":
        param_representation = layers
    else:
        param_representation = sum((list(layer.parameters()) for layer in layers), [])

    return cls(param_representation, **args_dict), args
