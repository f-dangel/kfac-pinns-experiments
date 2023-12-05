"""Verify the Jacobian structure of `W ↦ Z:= Wᵀ X W` with a matrix `X`.

Summary:
    1. The Jacobian `∂Z / ∂W = ∂vec(Z) / ∂vec(W)` is given by
        `∂Z / ∂W = I ⊗ (Wᵀ X) + K (I ⊗ (Wᵀ X))ᵀ` where `⊗` denotes the Kronecker
        product and `K` is the permutation matrix that acts on a row-space formed
        by the indices `(m, n)` of `W` and applies the re-ordering `(m, n) ↦ (n, m)`.

    2. The matrix-Jacobian product with a symmetric matrix `V` AND a symmetric `X`
       (as is the case for our application) simplifies to
       `V @ (∂Z / ∂W) = V @ (2 I ⊗ (Wᵀ X))` because the permutation matrix `K` has
       no effect on the symmetric matrix `V`.

Intro:
    Let `Z = Wᵀ X W`. We want to compute the Jacobian `∂Z / ∂W = ∂vec(Z) / ∂vec(W)`
    where `vec` denotes flattening according to first-index-varies-fastest
    (column-major). Note that this flattening convention is standard in the
    literature, but not in PyTorch which uses last-index-varies-fastest (row-major).

    This is annoying, because we need to do the calculations in both flattening
    conventions. However, to keep the presentation connected with previous works,
    it will make sense to use the standard convention prefered by the literature.
"""

from functools import partial
from itertools import product
from typing import Tuple

from einops import einsum, rearrange
from torch import Tensor, allclose, eye, kron, manual_seed, rand
from torch.func import jacrev


def vectorise(mat: Tensor) -> Tensor:
    """Flatten a matrix according to the literature convention (first-varies-fastest).

    Args:
        mat (Tensor): A matrix of shape `(M, N)`.

    Returns:
        Tensor: A vector of shape `(M * N)` corresponding to the flattened matrix.
            The elements are arranged as
            `(m=0, n=0), (1, 0), ..., (M-1, 0), (0, 1), ... (M-1, N-1))`
    """
    assert mat.ndim == 2
    return mat.T.flatten()


def matricise(vec: Tensor, shape: Tuple[int, int]) -> Tensor:
    """Unflatten a vector according to the literature convention (first-varies-fastest).

    Args:
        vec: A vector of shape `(M * N)`.
        shape: The shape of the resulting matrix.

    Returns:
        Tensor: A matrix of shape `(M, N)` corresponding to the unflattened vector.
            The elements `(0, 1, ..., MN-1)` of the vector are arranged as
            `0   M     ... (N-1)M
             1   M+1   ... (N-1)M+1
             ... ...   ...   ...
             M-1 2M-1  ... (N-1)M+M-1`
    """
    assert vec.ndim == 1
    return vec.reshape(shape[::-1]).T


def permute_grouped_indices(mat: Tensor, grouped_dims: Tuple[int, int]) -> Tensor:
    """Modify the order of indices that form the rows of `mat`.

    Args:
        mat: A matrix of shape `(M * N, L)` where dimensions `M, N` are grouped into
            one such that the row indices `m, n` are first-varies-fastest, i.e.
            `(m=0, n=0), (1, 0), ..., (M-1, 0), (0, 1), ... (M-1, N-1))`.
        grouped_dims: A tuple of the grouped dimensions `(M, N)`.

    Returns:
        Tensor: A matrix of shape `(N * M, L)` where the grouped dimensions are
            permuted such that the respective indices `n, m` are first-varies-fastest,
            i.e. `(n=0, m=0), (1, 0), ..., (N-1, 0), (0, 1), ... (N-1, M-1))`.
    """
    m, n = grouped_dims
    return rearrange(mat, "(m n) cols -> (n m) cols", n=n, m=m)


def f(W_flat: Tensor, X: Tensor, W_shape: Tuple[int, int]) -> Tensor:
    """Implement mapping `vec(W) ↦ vec(Wᵀ X W)`.

    Args:
        W_flat: Flattened weight matrix.
        X: Input matrix.
        W_shape: Shape of the weight matrix.

    Returns:
        Tensor: Flattened output matrix.
    """
    W = matricise(W_flat, W_shape)
    Z = W.T @ X @ W
    return vectorise(Z)


def jac_f_manual(W: Tensor, X: Tensor) -> Tensor:
    """Compute the Jacobian of `vec(W) ↦ vec(Wᵀ X W)` w.r.t. `vec(W)` manually.

    Args:
        W: Weight matrix.
        X: Input matrix.

    Returns:
        Tensor: The Jacobian of `vec(W) ↦ vec(Z := Wᵀ X W)` w.r.t. `vec(W)`.
             Has shape `(dim(vec(Z)), dim(vec(W)))`.
    """
    _, D_in = W.shape
    I_in = eye(D_in)
    return kron(I_in, W.T @ X) + permute_grouped_indices(
        kron(I_in, W.T @ X.T), (D_in, D_in)
    )


def mjp_f_manual(
    W: Tensor, X: Tensor, V: Tensor, X_symmetric: bool, V_symmetric: bool
) -> Tensor:
    """Compute the matrix-Jacobian product of `vec(W) ↦ vec(Wᵀ X W)` manually.

    Args:
        W: Weight matrix.
        X: Input matrix.
        V: Matrix to multiply the Jacobian with.
        X_symmetric: Whether `X` is symmetric.
        V_symmetric: Whether `V` is symmetric.

    Returns:
        Tensor: The matrix-Jacobian product of `vec(W) ↦ vec(Z := Wᵀ X W)` w.r.t.
            `vec(W)` with `V`. Has shape `(dim(vec(W)), dim(vec(Z)))`.
    """
    _, D_in = W.shape
    I_in = eye(D_in)
    I_kron_WTXT = kron(I_in, W.T @ X)

    first_term = einsum(V, I_kron_WTXT, "rows cols, rows w_flat -> w_flat cols")
    if X_symmetric and V_symmetric:  # this is the case we consider in the paper
        second_term = first_term
    else:  # this is the general case
        second_term = einsum(
            V,
            permute_grouped_indices(kron(I_in, W.T @ X.T), (D_in, D_in)),
            "rows cols, rows w_flat -> w_flat cols",
        )

    return first_term + second_term


def main():
    """Compare manual/automatic Jacobians and MJPs of `vec(W) ↦ vec(Wᵀ X W)`."""
    manual_seed(0)

    D_in = 3
    D_out = 2

    W_flat = rand(D_out * D_in)
    W = matricise(W_flat, (D_out, D_in))

    for X_symmetric, V_symmetric in product([False, True], repeat=2):
        print(f"X_symmetric={X_symmetric}, V_symmetric={V_symmetric}:")

        X = rand(D_out, D_out)
        if X_symmetric:
            X = 0.5 * (X + X.T)

        V = rand(D_in, D_in, D_in, D_in)
        if V_symmetric:
            V = 0.5 * (V + rearrange(V, "i j k l -> j i k l"))
            V = 0.5 * (V + rearrange(V, "i j k l -> i j l k"))
        V = V.reshape(D_in**2, D_in**2)

        # compute the Jacobian with autodiff and manually
        f_W = partial(f, X=X, W_shape=(D_out, D_in))
        jacobian_auto = jacrev(f_W)(W_flat)
        jacobian_manual = jac_f_manual(W, X)
        assert allclose(jacobian_auto, jacobian_manual)
        print("  Jacobians OK")

        # compute the matrix-Jacobian product with autodiff and manually
        mjp_auto = einsum(V, jacobian_auto, "rows cols, rows w_flat -> w_flat cols")
        mjp_manual = mjp_f_manual(W, X, V, X_symmetric, V_symmetric)
        assert allclose(mjp_auto, mjp_manual)
        print("  MJPs OK")


if __name__ == "__main__":
    main()
