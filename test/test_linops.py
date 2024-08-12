"""Test linear operators."""

from test.utils import report_nonclose

from pytest import mark
from torch import block_diag, manual_seed, rand
from torch.nn import Linear, Sequential, Tanh

from kfac_pinns_exp.gramian_utils import autograd_gramian
from kfac_pinns_exp.linops import GramianLinearOperator
from kfac_pinns_exp.train import create_condition_data, create_interior_data

EQUATIONS = [
    "poisson",
    "heat",
    "fokker-planck-isotropic",
    "log-fokker-planck-isotropic",
]
EQUATION_IDS = [f"equation={eq}" for eq in EQUATIONS]

APPROXIMATIONS = ["full", "per_layer"]
APPROXIMATION_IDS = [f"approximation={approx}" for approx in APPROXIMATIONS]

LOSS_TYPES = ["boundary", "interior"]
LOSS_TYPE_IDS = [f"loss_type={loss}" for loss in LOSS_TYPES]


@mark.parametrize("equation", EQUATIONS, ids=EQUATION_IDS)
@mark.parametrize("approximation", APPROXIMATIONS, ids=APPROXIMATION_IDS)
@mark.parametrize("loss_type", LOSS_TYPES, ids=LOSS_TYPE_IDS)
def test_GramianLinearOperator(equation: str, approximation: str, loss_type: str):
    """Check multiplication with the Gramian via pre-computed quantities.

    Args:
        equation: A string specifying the PDE.
        approximation: A string specifying the approximation of the Gramian.
        loss_type: A string specifying the type of loss.
    """
    manual_seed(0)
    dim_Omega = 2
    net_in_dim = {
        "poisson": dim_Omega,
        "heat": dim_Omega + 1,
        "fokker-planck-isotropic": dim_Omega + 1,
        "log-fokker-planck-isotropic": dim_Omega + 1,
    }[equation]
    layers = [Linear(net_in_dim, 4), Tanh(), Linear(4, 3), Tanh(), Linear(3, 1)]
    model = Sequential(*layers)
    N = 10
    data_fn = {
        "boundary": create_condition_data,
        "interior": create_interior_data,
    }[loss_type]
    boundary_condition = {
        "poisson": "sin_product",
        "heat": "sin_product",
        "fokker-planck-isotropic": "gaussian",
        "log-fokker-planck-isotropic": "gaussian",
    }[equation]
    X, y = data_fn(equation, boundary_condition, dim_Omega, N)

    # generate random vector
    num_params = sum(p.numel() for p in model.parameters())
    v = rand(num_params)

    # autodiff
    param_names = [n for n, _ in model.named_parameters()]
    gramian = autograd_gramian(
        model,
        X,
        param_names,
        loss_type=f"{equation}_{loss_type}",
        approximation=approximation,
    )
    if approximation == "per_layer":
        gramian = block_diag(*gramian)
    gramian.div_(N)
    Gv = gramian @ v

    # manual
    G_linop = GramianLinearOperator(
        equation, layers, X, y, loss_type, approximation=approximation
    )
    G_linop_v = G_linop @ v

    report_nonclose(Gv, G_linop_v)
