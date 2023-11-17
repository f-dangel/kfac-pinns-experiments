"""Tests for `kfac_pinns_exp.manual_differentiation`."""

from typing import Dict

from pytest import mark
from torch import manual_seed, rand
from torch.nn import Linear, Sequential, Sigmoid

from kfac_pinns_exp.manual_differentiation import manual_forward

CASES = [
    {
        "layers_fn": lambda: [Linear(10, 5), Sigmoid(), Linear(5, 3)],
        "input_fn": lambda: rand(4, 10),
        "seed": 0,
        "id": "linear-sigmoid-linear",
    },
]
CASE_IDS = [case["id"] for case in CASES]


@mark.parametrize("case", CASES, ids=CASE_IDS)
def test_manual_forward(case: Dict):
    """Test manual execution of a forward pass.

    Only checks for same output. Does not check the intermediate representations.

    Args:
        case: A dictionary describing a test case.
    """
    manual_seed(case["seed"])
    layers = case["layers_fn"]()
    X = case["input_fn"]()

    true_outputs = Sequential(*layers)(X)

    activations = manual_forward(layers, X)
    assert len(activations) == len(layers) + 1

    outputs = activations[-1]
    assert outputs.shape == true_outputs.shape
