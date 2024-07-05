"""Diffusivity, vector fields, and solutions of an isotropic Fokker-Planck equation."""

from torch import Tensor, linspace, meshgrid

from kfac_pinns_exp import fokker_planck_isotropic_equation

mu_isotropic = fokker_planck_isotropic_equation.mu_isotropic
sigma_isotropic = fokker_planck_isotropic_equation.sigma_isotropic


def q_isotropic_gaussian(X: Tensor) -> Tensor:
    """Isotropic Gaussian solution to the Fokker-Planck equation in log-space.

    Args:
        X: Batched quadrature points of shape `(N, d_Omega + 1)`.

    Returns:
        The function values as tensor of shape `(N, 1)`.
    """
    # TODO Implement this in log space to avoid numerical instabilities
    return fokker_planck_isotropic_equation.p_isotropic_gaussian(X).log()
