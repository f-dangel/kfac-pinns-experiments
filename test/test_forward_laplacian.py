"""Test `kfac_pinns_exp.forward_laplacian`."""

from test.test_manual_differentiation import CASE_IDS, CASES, set_up
from typing import Dict

from einops import einsum
from pytest import mark
from torch import allclose
from torch.nn import Sequential

from kfac_pinns_exp.autodiff_utils import autograd_input_hessian
from kfac_pinns_exp.forward_laplacian import manual_forward_laplacian


@mark.parametrize("case", CASES, ids=CASE_IDS)
def test_manual_forward_laplacian(case: Dict):
    """Compute the forward Laplacian and compare with functorch.

    Args:
        case: A dictionary describing a test case.
    """
    layers, X = set_up(case)

    # automatic computation (via functorch)
    true_hessian_X = autograd_input_hessian(Sequential(*layers), X)
    true_laplacian_X = einsum(true_hessian_X, "batch d d -> ")

    # forward-Laplacian computation
    coefficients = manual_forward_laplacian(layers, X)
    laplacian_X = einsum(coefficients[-1]["laplacian"], "n d -> ")

    assert allclose(true_laplacian_X, laplacian_X)
