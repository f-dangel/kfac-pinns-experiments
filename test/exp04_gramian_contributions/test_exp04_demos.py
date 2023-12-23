"""Run demos from exp04."""

from kfac_pinns_exp.exp04_gramian_contributions import (
    demo_gramian_contributions,
    demo_visualize_contributions,
)


def test_demo_gramian_contributions():
    """Execute the demo that computes the Gramian contributions."""
    demo_gramian_contributions.main()


def test_demo_visualize_contributions():
    """Execute the demo that visualizes the Gramian contributions."""
    demo_visualize_contributions.main()
