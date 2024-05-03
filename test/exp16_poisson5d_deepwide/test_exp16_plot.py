"""Run the plotting script from exp16."""

from kfac_pinns_exp.exp16_poisson5d_deepwide import plot
from kfac_pinns_exp.utils import run_verbose


def test_execute_plot():
    """Execute the plotting script."""
    cmd = ["python", plot.__file__, "--local_files", "--disable_tex"]
    run_verbose(cmd)
