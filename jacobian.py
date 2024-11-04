"""Attempt to efficiently compute Jacobians (needed for SPRING)."""

from time import time

from torch import allclose, cat, eye, manual_seed, rand, zeros
from torch.autograd import grad
from torch.nn import Linear, Sequential, Sigmoid

D_in, D_hidden, D_out = 100, 10, 1
assert D_out == 1
N1 = 6000
N2 = 4000
N = N1 + N2

manual_seed(0)

X1 = rand(N1, D_in)
X2 = rand(N2, D_in)

net = Sequential(
    Linear(D_in, D_hidden),
    Sigmoid(),
    Linear(D_hidden, D_out),
)

params = list(net.parameters())
P = sum(p.numel() for p in params)

# closure: X -> r(θ) ∈ Rᴺ, L(θ)
# compute: J_θ r ∈ Rᴺˣᴾ
r1 = net(X1)
r2 = net(X2)
r = cat([r1, r2])

t = time()

# naive
J = zeros(N, P)

for n in range(N):
    # n-th row of the Jacobian in list format
    J_n = grad(r[n], params, retain_graph=True)
    J_n = cat([j.flatten() for j in J_n])
    J[n, :] = J_n

print(f"Naive Jacobian took: {time() - t} s.")

# 'efficient'
t = time()
grad_outputs = eye(N).unsqueeze(-1)
J_fast = grad(r, params, grad_outputs=grad_outputs, is_grads_batched=True)
J_fast = cat([j.flatten(start_dim=1) for j in J_fast], dim=1)

print(f"Fast Jacobian took: {time() - t} s.")

assert allclose(J, J_fast)
