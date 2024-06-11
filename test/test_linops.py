"""Test linear operators."""

from test.utils import report_nonclose

from pytest import mark
from torch import manual_seed, rand
from torch.nn import Linear, Sequential, Tanh

from kfac_pinns_exp import heat_equation, poisson_equation
from kfac_pinns_exp.autodiff_utils import autograd_gramian
from kfac_pinns_exp.linops import BoundaryGramianLinearOperator
from kfac_pinns_exp.poisson_equation import square_boundary

EQUATIONS = ["poisson", "heat"]
EQUATION_IDS = [f"equation={eq}" for eq in EQUATIONS]


@mark.parametrize("equation", EQUATIONS, ids=EQUATION_IDS)
def test_BoundaryGramianLinearOperator(equation: str):
    """Check multiplication with the boundary Gramian via pre-computed quantities.

    Args:
        equation: A string specifying the PDE.
    """
    manual_seed(0)
    dim_Omega = {"poisson": 2, "heat": 3}[equation]
    layers = [Linear(dim_Omega, 4), Tanh(), Linear(4, 3), Tanh(), Linear(3, 1)]
    model = Sequential(*layers)
    N = 10
    X = square_boundary(N, dim_Omega)
    y = {
        "poisson": poisson_equation.u_sin_product(X),
        "heat": heat_equation.u_sin_product(X),
    }[equation]

    # generate random vector
    num_params = sum(p.numel() for p in model.parameters())
    v = rand(num_params)

    # autodiff
    param_names = [n for n, _ in model.named_parameters()]
    gramian = autograd_gramian(
        model, X, param_names, loss_type=f"{equation}_boundary"
    ).div_(N)
    Gv = gramian @ v

    # manual
    G_linop = BoundaryGramianLinearOperator(equation, layers, X, y)
    G_linop_v = G_linop @ v

    report_nonclose(Gv, G_linop_v)
