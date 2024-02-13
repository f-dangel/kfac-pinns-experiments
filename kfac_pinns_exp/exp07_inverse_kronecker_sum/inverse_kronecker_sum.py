"""Implements a class to multiply with the inverse of a sum of Kronecker matrices."""

from einops import einsum
from scipy.linalg import eigh
from torch import Tensor, dtype, float64, from_numpy, inverse


class InverseKroneckerSum:
    """Class to multiply with the inverse of the sum of two Kronecker products."""

    def __init__(
        self, A1: Tensor, A2: Tensor, B1: Tensor, B2: Tensor, inv_dtype: dtype = float64
    ):
        """Invert A₁ ⊗ A₂ + B₁ ⊗ B₂.

        A₁ and B₁ must have the same dimension.
        A₂ and B₂ must have the same dimension.
        All matrices must be symmetric positive-definite.

        Uses the relation
        (A₁ ⊗ A₂ + B₁ ⊗ B₂)⁻¹ = (V₁ ⊗ V₂) (Λ₁ ⊗ Λ₂ + I)⁻¹ (V₁⁻¹ B₁⁻¹ ⊗ V₂⁻¹ B₂⁻¹).
        where Vᵢ, Λᵢ are the solutions of the generalized eigenvalue problem
        Aᵢ Vᵢ = Bᵢ Vᵢ Λᵢ
        and Λᵢ is a diagonal matrix.

        Note:
            There is currently no PyTorch interface for solving generalized eigenvalue
            problems. This implementation uses the SciPy implementation, which costs
            GPU-to-CPU and CPU-to-GPU transfers if the tensors are on GPU.

        Args:
            A1: First matrix in the first Kronecker product.
            A2: Second matrix in the first Kronecker product.
            B1: First matrix in the second Kronecker product.
            B2: Second matrix in the second Kronecker product.
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
        if any(t.ndim != 2 or t.shape[0] != t.shape[1] for t in (A1, A2, B1, B2)):
            raise ValueError("All tensors must be 2d square.")
        if any(t.dtype != A1.dtype for t in (A2, B1, B2)):
            raise ValueError("All tensors must have the same data type.")
        if any(t.device != A1.device for t in (A2, B1, B2)):
            raise ValueError("All tensors must be on the same device.")
        if A1.shape != B1.shape or A2.shape != B2.shape:
            raise ValueError(
                "First and second Kronecker factors must match shapes. "
                + f"Got {A1.shape} vs {B1.shape}, {A2.shape} vs {B2.shape}."
            )

        dt = A1.dtype
        dev = A1.device
        self.kronecker_dims = (A1.shape[0], A2.shape[0])

        # solve generalized eigenvalue problem in SciPy in specified precision
        diagLam1, V1 = eigh(
            A1.cpu().to(inv_dtype).numpy(), B1.cpu().to(inv_dtype).numpy()
        )
        diagLam2, V2 = eigh(
            A2.cpu().to(inv_dtype).numpy(), B2.cpu().to(inv_dtype).numpy()
        )

        self.diagLam1 = from_numpy(diagLam1).to(dt).to(dev)
        self.diagLam2 = from_numpy(diagLam2).to(dt).to(dev)

        # compute required inverses in specified precision, store in original precision
        V1 = from_numpy(V1).to(dev).to(inv_dtype)
        V2 = from_numpy(V2).to(dev).to(inv_dtype)

        B1_inv = inverse(B1.to(inv_dtype)).to(dt)
        V1_inv = inverse(V1).to(dt)
        self.V1_inv_B1_inv = V1_inv @ B1_inv

        B2_inv = inverse(B2.to(inv_dtype)).to(dt)
        V2_inv = inverse(V2).to(dt)
        self.B2_inv_T_V2_inv_T = (V2_inv @ B2_inv).T

        self.V1 = V1.to(dt)
        self.V2_T = V2.to(dt).T

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

        x = self.V1_inv_B1_inv @ x @ self.B2_inv_T_V2_inv_T
        x = x.flatten() / (
            einsum(self.diagLam1, self.diagLam2, "i, j -> i j").flatten() + 1
        )
        x = x.reshape(*self.kronecker_dims)
        x = self.V1 @ x @ self.V2_T

        return x.flatten() if flattened else x
