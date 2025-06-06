"""Tests for `kfac_pinns_exp.manual_differentiation`."""

from typing import Dict, List, Tuple

from pytest import mark
from torch import Tensor, allclose, manual_seed, rand, rand_like
from torch.autograd import grad
from torch.nn import Linear, Module, ReLU, Sequential, Sigmoid, Tanh

from kfac_pinns_exp.autodiff_utils import autograd_input_hessian
from kfac_pinns_exp.manual_differentiation import (
    manual_backward,
    manual_forward,
    manual_hessian_backward,
)

CASES = [
    {
        # NOTE The last layer should produce a single number
        "layers_fn": lambda: [Linear(10, 5), Sigmoid(), Linear(5, 1)],
        "input_fn": lambda: rand(4, 10),
        "seed": 0,
        "id": "linear-sigmoid-linear",
    },
    {
        "layers_fn": lambda: [
            Linear(10, 8),
            Sigmoid(),
            Linear(8, 6),
            Sigmoid(),
            Linear(6, 4),
            Sigmoid(),
            Linear(4, 1),
        ],
        "input_fn": lambda: rand(4, 10),
        "seed": 0,
        "id": "3x(linear-sigmoid)-linear",
    },
    {
        "layers_fn": lambda: [Linear(10, 5), Tanh(), Linear(5, 1)],
        "input_fn": lambda: rand(4, 10),
        "seed": 0,
        "id": "linear-tanh-linear",
    },
    {
        "layers_fn": lambda: [
            Linear(10, 8),
            Tanh(),
            Linear(8, 6),
            Tanh(),
            Linear(6, 4),
            Tanh(),
            Linear(4, 1),
        ],
        "input_fn": lambda: rand(4, 10),
        "seed": 0,
        "id": "3x(linear-tanh)-linear",
    },
    {
        "layers_fn": lambda: [
            Linear(10, 8),
            ReLU(),
            Linear(8, 6),
            ReLU(),
            Linear(6, 4),
            ReLU(),
            Linear(4, 1),
        ],
        "input_fn": lambda: rand(4, 10),
        "seed": 0,
        "id": "3x(linear-relu)-linear",
    },
    {
        "layers_fn": lambda: [
            Linear(10, 8),
            ReLU(),
            Linear(8, 6),
            Sigmoid(),  # needed to introduce some curvature
            Linear(6, 4),
            Tanh(),  # needed to introduce some curvature
            Linear(4, 1),
        ],
        "input_fn": lambda: rand(4, 10),
        "seed": 0,
        "id": "mixed",
    },
]
CASE_IDS = [case["id"] for case in CASES]


def set_up(case: Dict) -> Tuple[List[Module], Tensor]:
    """Set random seed and instantiate a case.

    Args:
        case: A dictionary describing a test case.

    Returns:
        A tuple of the instantiated layers and input.
    """
    manual_seed(case["seed"])
    layers = case["layers_fn"]()
    X = case["input_fn"]()
    return layers, X


@mark.parametrize("case", CASES, ids=CASE_IDS)
def test_manual_forward(case: Dict):
    """Test manual execution of a forward pass.

    Only checks for same output. Does not check the intermediate representations.

    Args:
        case: A dictionary describing a test case.
    """
    layers, X = set_up(case)

    true_outputs = Sequential(*layers)(X)

    activations = manual_forward(layers, X)
    assert len(activations) == len(layers) + 1
    assert allclose(activations[-1], true_outputs)


@mark.parametrize("case", CASES, ids=CASE_IDS)
def test_manual_backward(case: Dict):
    """Test manual execution of a backward pass.

    Only checks for same gradient w.r.t. the first input. Does not check the
    intermediate gradients.

    Args:
        case: A dictionary describing a test case.
    """
    layers, X = set_up(case)
    X.requires_grad_(True)  # to compute gradients w.r.t. X

    true_outputs = Sequential(*layers)(X)
    grad_outputs = rand_like(true_outputs)
    true_grad_X = grad(true_outputs, X, grad_outputs=grad_outputs)[0]

    activations = manual_forward(layers, X)
    gradients = manual_backward(layers, activations, grad_outputs=grad_outputs)
    assert len(gradients) == len(layers) + 1
    assert allclose(gradients[0], true_grad_X)


@mark.parametrize("case", CASES, ids=CASE_IDS)
def test_manual_hessian_backward(case: Dict):
    """Test manual execution of a Hessian backward pass.

    Only checks for same Hessians w.r.t. the first input. Does not check the
    intermediate Hessians.

    Args:
        case: A dictionary describing a test case.
    """
    layers, X = set_up(case)

    true_hess_X = autograd_input_hessian(Sequential(*layers), X)

    activations = manual_forward(layers, X)
    gradients = manual_backward(layers, activations)
    hessians = manual_hessian_backward(layers, activations, gradients)

    assert allclose(true_hess_X, hessians[0])
