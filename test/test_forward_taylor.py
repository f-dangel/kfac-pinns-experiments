"""Test `kfac_pinns_exp.forward_taylor`."""

from test.test_manual_differentiation import CASE_IDS, CASES, set_up
from test.utils import report_nonclose
from typing import Dict

from einops import rearrange
from pytest import mark
from torch.nn import Sequential

from kfac_pinns_exp.autodiff_utils import (
    autograd_input_hessian,
    autograd_input_jacobian,
)
from kfac_pinns_exp.forward_taylor import manual_forward_taylor


@mark.parametrize("case", CASES, ids=CASE_IDS)
def test_manual_forward_taylor(case: Dict):
    """Compute the Taylor coefficients and compare with functorch.

    Args:
        case: A dictionary describing a test case.
    """
    layers, X = set_up(case)

    # automatic computation (via functorch)
    model = Sequential(*layers)
    true_output = model(X)
    true_jac_X = autograd_input_jacobian(model, X)
    true_hessian_X = autograd_input_hessian(model, X)

    # forward-Laplacian computation
    coefficients = manual_forward_taylor(layers, X)
    output = coefficients[-1]["c_0"]
    # Taylor-mode uses different shape convention than autograd for Jacobian and Hessian
    jac_X = rearrange(coefficients[-1]["c_1"], "batch d_0 d_out -> batch d_out d_0")
    hessian_X = coefficients[-1]["c_2"].squeeze(-1)

    report_nonclose(true_output, output)
    report_nonclose(true_jac_X, jac_X)
    report_nonclose(true_hessian_X, hessian_X)
