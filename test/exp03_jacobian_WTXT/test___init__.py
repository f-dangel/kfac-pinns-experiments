"""Execute the experiment that investigates the Jacobian of `vec(W) ↦ vec(Wᵀ X W)`."""

from kfac_pinns_exp.exp03_jacobian_WTXT import main


def test_exp03__init__():
    """Execute the `__init__` file's main function."""
    main()
