"""Verify the optimal approximation of `vec(G) @ vec(G)ᵀ` via `I_S ⊗ U`.

The optimal solution is `U = G @ Gᵀ / S`.
"""

from torch import Tensor, allclose, einsum, eye, kron, manual_seed, rand
from torch.optim import LBFGS


def main():
    """Verify the optimal approximation of `vec(G) @ vec(G)ᵀ` via `I_S ⊗ U`.

    Raises:
        AssertionError: If the solution is not close enough to the expected one.
    """
    # setup
    manual_seed(0)
    D_out, S = 10, 5

    G = rand(D_out, S)
    vec_G = G.T.flatten()  # column-stacking, like in the literature
    ggT = einsum("i,j->ij", vec_G, vec_G)

    I = eye(S)
    U = rand(D_out, D_out, requires_grad=True)
    U_opt = (G @ G.T) / S  # expected solution

    def closure() -> Tensor:
        """Set gradients to zero, evaluate the objective, and backpropagate through it.

        Returns:
            The objective value.
        """
        optimizer.zero_grad()
        error = ((kron(I, U) - ggT) ** 2).sum()
        error.backward()
        return error

    # set up the optimizer and optimize
    optimizer = LBFGS([U])

    num_steps = 3
    for step in range(num_steps):
        print(f"Step {step:03g}")
        print(
            "\t||U - U_opt||_F^2 / ||U_opt||_F^2 = "
            + f"{(U - U_opt).norm()**2 / U_opt.norm()**2:.2e}"
        )
        error = optimizer.step(closure=closure)
        print(f"\tReconstruction error: {error:.2e}")

    # verify the solution
    tols = {"rtol": 1e-4}
    mismatches = 0
    for u1, u2 in zip(U.flatten(), U_opt.flatten()):
        if not allclose(u1, u2, **tols):
            print(f"{u1} ≠ {u2}, ratio {u1 / u2}")
            mismatches += 1
    print(f"Total mismatches: {mismatches} / {U.numel()}")
    assert allclose(U, U_opt, **tols)


if __name__ == "__main__":
    main()
