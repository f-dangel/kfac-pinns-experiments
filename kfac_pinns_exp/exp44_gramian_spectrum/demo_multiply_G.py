"""Demonstrate how to multiply with G."""

from time import time

from backpack.hessianfree.ggnvp import ggn_vector_product_from_plist
from torch import allclose, cat, device, float64, manual_seed, rand_like, zeros
from torch.linalg import eigvalsh
from torch.nn import Sequential

from kfac_pinns_exp.linops import GramianLinearOperator
from kfac_pinns_exp.train import (
    INTERIOR_LOSS_EVALUATORS,
    create_condition_data,
    create_interior_data,
    evaluate_boundary_loss,
    set_up_layers,
)

# Parameters to control the problem
model_name = "mlp-tanh-64"
equation = "poisson"
condition = "sin_product"
dim_Omega = 2
num_data = 128
loss_type = "interior"  # alternative value: "boundary"
dt, dev = float64, device("cpu")

# make deterministic
manual_seed(0)

# set up the neural network
layers = [layer.to(dev, dt) for layer in set_up_layers(model_name, equation, dim_Omega)]
net = Sequential(*layers).to(dev, dt)

# set up the data
data_generator = {"interior": create_interior_data, "boundary": create_condition_data}[
    loss_type
]
X, y = data_generator(equation, condition, dim_Omega, num_data)
X, y = X.to(dev, dt), y.to(dev, dt)

# get the function to evaluate the residual and loss
loss_evaluator = {
    "interior": INTERIOR_LOSS_EVALUATORS[equation],
    "boundary": evaluate_boundary_loss,
}[loss_type]

# Compute the loss and residual. We have different ways of doing that. Let's compare:

# 1) Passing the NN. This uses `functorch` to compute differential operators (slow)
start = time()
loss1, residual1, _ = loss_evaluator(net, X, y)
print(f"Loss and residual via functorch: {time() - start:.3g} s")

# 2) Passing layers. This uses Taylor-mode to compute differential operators (fast)
start = time()
loss2, residual2, _ = loss_evaluator(layers, X, y)
print(f"Loss and residual via Taylor-mode: {time() - start:.3g} s")

assert allclose(loss1, loss2)
assert allclose(residual1, residual2)
print("Functorch = Taylor-mode")

# Generate a random vector v in parameter list format
v = [rand_like(p) for p in net.parameters()]

# Multiply the Gramian onto v, get the result in parameter list format
params = [p for p in net.parameters() if p.requires_grad]
Gv = ggn_vector_product_from_plist(loss2, residual2, params, v)

# Now that we now how to multiply with G, we can compute G's matrix representation
Ds = [p.numel() for p in params]
D = sum(Ds)

start = time()
G = zeros(D, D, dtype=dt, device=dev)

for d in range(D):
    # create the standard vector along direction d
    e_d = zeros(D, dtype=dt, device=dev)
    e_d[d] = 1.0

    # convert into list format
    e_d = e_d.split(Ds)
    e_d = [e.reshape_as(p) for e, p in zip(e_d, params)]

    # multiply by G, get result in list format
    Ge_d = ggn_vector_product_from_plist(loss2, residual2, params, e_d)

    # convert to vector, this is the d-th column of G
    Ge_d = cat([e.flatten() for e in Ge_d])

    G[:, d] = Ge_d

print(f"Gramian computation took {time() - start:.3f} s.")
print(f"Gramian:\n{G}")

# Let's compute the eigenvalues of G
evals = eigvalsh(G)
print(f"dim(G) = {D}")
print(f"Naive rank bound: rank(G) <= {num_data}.")
threshold = 1e-10
print(f"Eigenvalues > {threshold}: {(evals > threshold).int().sum()}")

# Fast Gramian-vector products
start = time()
# TODO Implement multiplication with multiple vectors at once to avoid for loop
G_linop = GramianLinearOperator(equation, layers, X, y, loss_type)

G2 = zeros(D, D, dtype=dt, device=dev)
for d in range(D):
    # create the standard vector along direction d
    e_d = zeros(D, dtype=dt, device=dev)
    e_d[d] = 1.0
    Ge_d = G_linop @ e_d
    G2[:, d] = Ge_d
print(f"Gramian computation with linop took {time() - start:.3f} s.")
assert allclose(G, G2)
