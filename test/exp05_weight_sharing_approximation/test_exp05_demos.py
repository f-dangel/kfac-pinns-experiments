"""Run demos from exp05."""

from kfac_pinns_exp.exp05_weight_sharing_approximation import demo_optimal_approximation


def test_demo_optimal_approximation():
    """Execute demo that verifies optimal approximation of `vec(G) @ vec(G)ᵀ≈I ⊗ U`."""
    demo_optimal_approximation.main()
