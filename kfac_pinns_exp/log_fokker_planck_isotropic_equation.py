"""Diffusivity, vector fields, and solutions of an isotropic Fokker-Planck equation."""

from functools import partial

from torch import Tensor

from kfac_pinns_exp import (
    fokker_planck_equation,
    fokker_planck_isotropic_equation,
    log_fokker_planck_equation,
)

mu_isotropic = fokker_planck_isotropic_equation.mu_isotropic
div_mu_isotropic = fokker_planck_isotropic_equation.div_mu_isotropic
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


evaluate_interior_loss = partial(
    log_fokker_planck_equation.evaluate_interior_loss,
    mu=mu_isotropic,
    sigma=sigma_isotropic,
    div_mu=div_mu_isotropic,
)

evaluate_interior_loss_and_kfac = partial(
    log_fokker_planck_equation.evaluate_interior_loss_and_kfac,
    mu=mu_isotropic,
    sigma=sigma_isotropic,
    div_mu=div_mu_isotropic,
)

plot_solution = partial(
    fokker_planck_equation.plot_solution, solutions={"gaussian": q_isotropic_gaussian}
)
