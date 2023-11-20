"""Run demos from exp01."""

from kfac_pinns_exp.exp01_manual_laplacian_and_fisher import (
    demo_gramian,
    demo_laplacian,
)


def test_demo_laplacian():
    """Execute the demo that computes the Laplacian."""
    demo_laplacian.main()


def test_demo_gramian():
    """Execute the demo that computes the Gramian."""
    demo_gramian.main()
