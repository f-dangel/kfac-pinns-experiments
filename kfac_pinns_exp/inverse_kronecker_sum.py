"""Implements a class to multiply with the inverse of a sum of Kronecker matrices."""

from typing import Tuple

from einops import einsum
from scipy.linalg import eigh as scipy_eigh
from torch import Tensor, dtype, float64, from_numpy, inverse
from torch.linalg import eigh as torch_eigh


class InverseKroneckerSum:
    """Class to multiply with the inverse of the sum of two Kronecker products."""

    def __init__(
        self,
        A1: Tensor,
        A2: Tensor,
        B1: Tensor,
        B2: Tensor,
        inv_dtype: dtype = float64,
        backend: str = "torch",
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

        Args:
            A1: First matrix in the first Kronecker product.
            A2: Second matrix in the first Kronecker product.
            B1: First matrix in the second Kronecker product.
            B2: Second matrix in the second Kronecker product.
            inv_dtype: Data type in which matrix inversions and eigen-decompositions
                are performed. Those operations are often unstable in low precision.
                Therefore, it is often helpful to carry them out in higher precision.
                Default is `float64`.
            backend: Backend to use for solving the generalized eigenvalue problem.
                Currently supports `"torch"` and `"scipy"`. Default is `"torch"`, which
                uses `scipy.linalg.eigh` which requires GPU-CPU syncs. `"torch"` will
                use a PyTorch implementation that is based on the description in
                ['Eigenvalue and Generalized Eigenvalue Problems:
                Tutorial'](https://arxiv.org/pdf/1903.11240.pdf) that might be
                numerically less stable but runs on GPU.

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

        (dt,) = {A1.dtype, A2.dtype, B1.dtype, B2.dtype}
        self.kronecker_dims = (A1.shape[0], A2.shape[0])

        # solve generalized eigenvalue problem in specified precision
        diagLam1, V1 = self.eigh(A1.to(inv_dtype), B1.to(inv_dtype), backend)
        diagLam2, V2 = self.eigh(A2.to(inv_dtype), B2.to(inv_dtype), backend)

        # convert eigenvalues back to original precision
        self.diagLam1 = diagLam1.to(dt)
        self.diagLam2 = diagLam2.to(dt)

        # compute the other required quantities in `inv_dtype` precision and convert
        # back to original precision
        B1_inv = inverse(B1.to(inv_dtype)).to(dt)
        V1_inv = inverse(V1).to(dt)
        self.V1_inv_B1_inv = V1_inv @ B1_inv

        B2_inv = inverse(B2.to(inv_dtype)).to(dt)
        V2_inv = inverse(V2).to(dt)
        self.B2_inv_T_V2_inv_T = (V2_inv @ B2_inv).T

        self.V1 = V1.to(dt)
        self.V2_T = V2.to(dt).T

    @classmethod
    def eigh(cls, A: Tensor, B: Tensor, backend: str) -> Tuple[Tensor, Tensor]:
        """Solve a generalized eigenvalue problem defined by (A, B).

        Finds ϕ (dxd, orthonormal), Λ (dxd, diagonal) such that A ϕ = B ϕ Λ.

        Args:
            A: Symmetric positive definite matrix.
            B: Symmetric positive definite matrix.

        Returns:
            Tuple of generalized eigenvalues (Λ) and eigenvectors (ϕ).

        Raises:
            ValueError: If the backend is not supported.
        """
        if backend == "scipy":
            (dev,) = {A.device, B.device}
            diagLam, V = scipy_eigh(A.cpu().numpy(), B.cpu().numpy())
            return from_numpy(diagLam).to(dev), from_numpy(V).to(dev)

        elif backend == "torch":
            # see Algorithm 1 in https://arxiv.org/pdf/1903.11240.pdf
            diagLam_B, Phi_B = torch_eigh(B)
            Phi_B = einsum(Phi_B, diagLam_B.pow(-0.5), "i j, j -> i j")
            diagLamA, Phi_A = torch_eigh(Phi_B.T @ A @ Phi_B)
            return diagLamA, Phi_B @ Phi_A

        raise ValueError(
            f"Unsupported backend: {backend}. Supported: 'torch', 'scipy'."
        )

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
