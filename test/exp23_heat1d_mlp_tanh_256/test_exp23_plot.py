"""Run the plotting script from exp23."""

from kfac_pinns_exp.exp23_heat1d_mlp_tanh_256 import plot
from kfac_pinns_exp.utils import run_verbose


def test_execute_plot():
    """Execute the plotting script."""
    cmd = ["python", plot.__file__, "--local_files", "--disable_tex"]
    run_verbose(cmd)
