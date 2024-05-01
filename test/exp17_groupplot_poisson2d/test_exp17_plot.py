"""Run the plotting script from exp17."""

from kfac_pinns_exp.exp17_groupplot_poisson2d import plot
from kfac_pinns_exp.utils import run_verbose


def test_execute_plot():
    """Execute the plotting script."""
    cmd = ["python", plot.__file__, "--disable_tex"]
    run_verbose(cmd)
