"""Test SPRING-related functionality."""

from torch import allclose, cuda, device, eye, float64, isclose, manual_seed
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

# make sure the eigenvalues of the Gramian and Jacobian outer product match

# hyper-parameters
equation = "poisson"
boundary_condition = "sin_product"
N_Omega = 64
N_dOmega = 32
dim_Omega = 2
model = "mlp-tanh-64"
dt = float64
dev = device("cuda" if cuda.is_available() else "cpu")

manual_seed(0)  # make deterministic

# generate neural network and data
layers = set_up_layers(model, equation, dim_Omega)
layers = [layer.to(dev, dt) for layer in layers]

X_Omega, y_Omega = create_interior_data(
    equation, boundary_condition, dim_Omega, N_Omega
)
X_Omega, y_Omega = X_Omega.to(dev, dt), y_Omega.to(dev, dt)
X_dOmega, y_dOmega = create_condition_data(
    equation, boundary_condition, dim_Omega, N_dOmega
)
X_dOmega, y_dOmega = X_dOmega.to(dev, dt), y_dOmega.to(dev, dt)

num_params = sum(sum(p.numel() for p in layer.parameters()) for layer in layers)

# ground truth: Eigenvalues of the Gramian
G_interior = GramianLinearOperator(equation, layers, X_Omega, y_Omega, "interior")
G_boundary = GramianLinearOperator(equation, layers, X_dOmega, y_dOmega, "boundary")

identity = eye(num_params, device=dev, dtype=dt)
G = G_interior @ identity + G_boundary @ identity
G_evals = eigvalsh(G)

# compare with: Eigenvalues of the Jacobian outer product
_, _, interior_inputs, interior_grad_outputs, boundary_inputs, boundary_grad_outputs = (
    evaluate_losses_with_layer_inputs_and_grad_outputs(
        layers, X_Omega, y_Omega, X_dOmega, y_dOmega, equation
    )
)
JJT = compute_jacobian_outer_product(
    interior_inputs, interior_grad_outputs, boundary_inputs, boundary_grad_outputs
)
JJT_evals = eigvalsh(JJT)

# clip to same length and sort descendingly
effective_evals = min(len(G_evals), len(JJT_evals))

G_evals = G_evals.flip(0)[:effective_evals]
JJT_evals = JJT_evals.flip(0)[:effective_evals]

# comparison
if not allclose(G_evals, JJT_evals):
    for idx, (e_G, e_JJT) in enumerate(zip(G_evals, JJT_evals)):
        print(
            f"{idx:03g}: {e_G.item():.5g} vs {e_JJT.item():.5g} "
            + f"-> {isclose(e_G, e_JJT).item()}"
        )
    raise ValueError(
        "Eigenvalues of the Gramian and Jacobian outer product do not match."
    )
