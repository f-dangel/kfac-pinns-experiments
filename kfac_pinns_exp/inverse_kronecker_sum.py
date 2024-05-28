"""Implements a class to multiply with the inverse of a sum of Kronecker matrices."""

from einops import einsum
from torch import Tensor, dtype, float64
from torch.linalg import eigh


class InverseKroneckerSum:
    """Class to multiply with the inverse of the sum of two Kronecker products."""

    def __init__(
        self, A: Tensor, B: Tensor, C: Tensor, D: Tensor, inv_dtype: dtype = float64
    ):
        """Invert A ⊗ B + C ⊗ D.

        A and C must have the same dimension.
        B and D must have the same dimension.
        All matrices must be symmetric positive-definite.
        See Martens 2015, Appendix B for details.

        Args:
            A: First matrix in the first Kronecker product.
            B: Second matrix in the first Kronecker product.
            C: First matrix in the second Kronecker product.
            D: Second matrix in the second Kronecker product.
            inv_dtype: Data type in which matrix inversions and eigen-decompositions
                are performed. Those operations are often unstable in low precision.
                Therefore, it is often helpful to carry them out in higher precision.
                Default is `float64`.

        Raises:
            ValueError: If first and second Kronecker factors don't match shapes.
            ValueError: If any of the tensors is not 2d.
            ValueError: If the tensors do not share the same data type.
            ValueError: If the tensors do not share the same device.
        """
        if any(t.ndim != 2 or t.shape[0] != t.shape[1] for t in (A, B, C, D)):
            raise ValueError("All tensors must be 2d square.")
        if any(t.dtype != A.dtype for t in (B, C, D)):
            raise ValueError("All tensors must have the same data type.")
        if any(t.device != A.device for t in (B, C, D)):
            raise ValueError("All tensors must be on the same device.")
        if A.shape != C.shape or B.shape != D.shape:
            raise ValueError(
                "First and second Kronecker factors must match shapes. "
                + f"Got {A.shape} vs {C.shape}, {B.shape} vs {D.shape}."
            )

        (dt,) = {A.dtype, B.dtype, C.dtype, D.dtype}
        self.kronecker_dims = (A.shape[0], B.shape[0])

        # compute A^{-1/2}
        A_evals, A_evecs = eigh(A.to(inv_dtype))
        A_inv_sqrt = (A_evecs / A_evals.sqrt()) @ A_evecs.T

        # compute B^{-1/2}
        B_evals, B_evecs = eigh(B.to(inv_dtype))
        B_inv_sqrt = (B_evecs / B_evals.sqrt()) @ B_evecs.T

        # compute E1, E2
        S1, E1 = eigh(A_inv_sqrt @ C.to(inv_dtype) @ A_inv_sqrt)
        S2, E2 = eigh(B_inv_sqrt @ D.to(inv_dtype) @ B_inv_sqrt)

        # compute K1, K2
        K1, K2 = A_inv_sqrt @ E1, B_inv_sqrt @ E2

        # convert back to original precision
        self.S1, self.S2 = S1.to(dt), S2.to(dt)
        self.K1, self.K2 = K1.to(dt), K2.to(dt)

    def __matmul__(self, x: Tensor) -> Tensor:
        """Multiply the inverse onto a vector (@ operator).

        Let D₁ denote the dimension of A₁ and D₂ the dimension of A₂.

        Args:
            x: Vector to multiply with the inverse. Can either be a vector of shape
                (D₁ * D₂) or a matrix reshape of size (D₁, D₂).

        Returns:
            The product of the inverse with the vector, either in form of a (D₁ * D₂)
            vector or a (D₁, D₂) matrix, depending on the input.

        Raises:
            ValueError: If the input is not of the correct shape.
        """
        total_dim = self.kronecker_dims[0] * self.kronecker_dims[1]
        if x.ndim == 2 and x.shape == self.kronecker_dims:
            flattened = False
        elif x.ndim == 1 and x.shape[0] == total_dim:
            flattened = True
        else:
            raise ValueError(
                f"Input must be {self.kronecker_dims} or {total_dim}. Got {x.shape}"
            )
        if flattened:
            x = x.reshape(*self.kronecker_dims)

        x = self.K1.T @ x @ self.K2
        x /= einsum(self.S1, self.S2, "i, j -> i j") + 1.0
        x = self.K1 @ x @ self.K2.T

        return x.flatten() if flattened else x
