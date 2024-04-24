"""Test `kfac_pinns_exp.forward_laplacian`."""

from test.test_manual_differentiation import CASE_IDS, CASES, set_up
from typing import Callable, Dict, List, Union

from einops import einsum
from pytest import mark
from torch import Tensor, allclose
from torch.nn import Sequential

from kfac_pinns_exp.autodiff_utils import autograd_input_hessian
from kfac_pinns_exp.forward_laplacian import manual_forward_laplacian

COORDINATE_FNS = {
    "coordinates=None": lambda X: None,
    "coordinates=even": lambda X: [i for i in range(X.shape[1]) if i % 2 == 0],
}


@mark.parametrize("coordinate_fn", COORDINATE_FNS.values(), ids=COORDINATE_FNS.keys())
@mark.parametrize("case", CASES, ids=CASE_IDS)
def test_manual_forward_laplacian(
    case: Dict, coordinate_fn: Callable[[Tensor], Union[None, List[int]]]
):
    """Compute the forward Laplacian and compare with functorch.

    Args:
        case: A dictionary describing a test case.
    """
    layers, X = set_up(case)
    coordinates = coordinate_fn(X)

    # automatic computation (via functorch)
    true_hessian_X = autograd_input_hessian(
        Sequential(*layers), X, coordinates=coordinates
    )
    if coordinates is not None:
        assert true_hessian_X.shape[1:] == (len(coordinates), len(coordinates))
    true_laplacian_X = einsum(true_hessian_X, "batch d d -> ")

    # forward-Laplacian computation
    coefficients = manual_forward_laplacian(layers, X, coordinates=coordinates)
    laplacian_X = einsum(coefficients[-1]["laplacian"], "n d -> ")

    assert allclose(true_laplacian_X, laplacian_X)
