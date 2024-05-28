"""Run the plotting script from exp18."""

from kfac_pinns_exp.exp18_groupplot_poisson5d import plot
from kfac_pinns_exp.utils import run_verbose


def test_execute_plot():
    """Execute the plotting script."""
    cmd = ["python", plot.__file__, "--disable_tex"]
    run_verbose(cmd)
