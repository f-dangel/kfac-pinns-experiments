"""Test line search methods."""

from numpy import isclose
from torch import Tensor

from kfac_pinns_exp.optim.line_search import grid_line_search


def test_grid_line_search():
    """Test the grid search method on a quadratic function."""
    params = [Tensor([0.5])]

    def f() -> Tensor:
        """A simple quadratic function.

        Returns:
            The squared value of `params[0]`.
        """
        return params[0] ** 2

    step = [Tensor([-1.0])]
    grid = [0.1, 0.3, 1.0]

    best_lr, best_f = grid_line_search(f, params, step, grid)

    assert best_lr == 0.3
    assert isclose(best_f, 0.04)
