"""Test SPRING-related functionality."""

from test.utils import DEVICE_IDS, DEVICES, report_nonclose
from typing import List, Tuple

from pytest import mark
from torch import device, dtype, eye, float64, manual_seed
from torch.linalg import eigvalsh

from kfac_pinns_exp.linops import GramianLinearOperator
from kfac_pinns_exp.optim.spring import (
    compute_jacobian_outer_product,
    evaluate_losses_with_layer_inputs_and_grad_outputs,
)
from kfac_pinns_exp.train import (
    create_condition_data,
    create_interior_data,
    set_up_layers,
)

PDE_CASES: List[Tuple[str, str]] = [
    ("poisson", "sin_product"),
    ("poisson", "cos_sum"),
    ("heat", "sin_product"),
    ("fokker-planck-isotropic", "gaussian"),
    ("log-fokker-planck-isotropic", "gaussian"),
]
PDE_IDS = [f"{equation}-{condition}" for equation, condition in PDE_CASES]
MODEL_CASES = ["mlp-tanh-64"]


@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
@mark.parametrize("model", MODEL_CASES, ids=MODEL_CASES)
@mark.parametrize("equation, condition", PDE_CASES, ids=PDE_IDS)
def test_leading_eigenvalues_G_and_JJT(
    equation: str,
    condition: str,
    model: str,
    device: device,
    dtype: dtype = float64,
    N_Omega: int = 64,
    N_dOmega: int = 32,
    dim_Omega: int = 3,
):
    """Make sure the leading Gramian and Jacobian outer product eigenvalues match.

    Args:
        equation: String specifying the PDE.
        condition: String specifying the conditions.
        model: String specifying the model.
        device: The device to run the test on.
        dtype: The data type to run the test in. Defaults to `float64`.
        N_Omega: The number of interior data points. Defaults to `64`.
        N_dOmega: The number of boundary data points. Defaults to `32`.
        dim_Omega: The dimension of the domain. Defaults to `3`.
    """
    manual_seed(0)  # make deterministic

    # generate neural network and data
    layers = [
        layer.to(device, dtype) for layer in set_up_layers(model, equation, dim_Omega)
    ]
    num_params = sum(sum(p.numel() for p in layer.parameters()) for layer in layers)

    X_Omega, y_Omega = [
        t.to(device, dtype)
        for t in create_interior_data(equation, condition, dim_Omega, N_Omega)
    ]
    X_dOmega, y_dOmega = [
        t.to(device, dtype)
        for t in create_condition_data(equation, condition, dim_Omega, N_dOmega)
    ]

    # ground truth: Eigenvalues of the Gramian
    G_interior = GramianLinearOperator(equation, layers, X_Omega, y_Omega, "interior")
    G_boundary = GramianLinearOperator(equation, layers, X_dOmega, y_dOmega, "boundary")

    identity = eye(num_params, device=device, dtype=dtype)
    G = G_interior @ identity + G_boundary @ identity
    G_evals = eigvalsh(G)

    # compare with: Eigenvalues of the Jacobian outer product
    (
        _,
        _,
        interior_inputs,
        interior_grad_outputs,
        boundary_inputs,
        boundary_grad_outputs,
    ) = evaluate_losses_with_layer_inputs_and_grad_outputs(
        layers, X_Omega, y_Omega, X_dOmega, y_dOmega, equation
    )
    JJT = compute_jacobian_outer_product(
        interior_inputs, interior_grad_outputs, boundary_inputs, boundary_grad_outputs
    )
    JJT_evals = eigvalsh(JJT)

    # clip to same length and sort descendingly
    effective_evals = min(num_params, N_Omega + N_dOmega)
    G_evals = G_evals.flip(0)[:effective_evals]
    JJT_evals = JJT_evals.flip(0)[:effective_evals]

    report_nonclose(G_evals, JJT_evals)
