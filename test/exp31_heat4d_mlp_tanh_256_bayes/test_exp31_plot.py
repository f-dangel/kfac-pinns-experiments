"""Run the plotting script from exp31."""

from kfac_pinns_exp.exp31_heat4d_mlp_tanh_256_bayes import plot
from kfac_pinns_exp.utils import run_verbose


def test_execute_plot():
    """Execute the plotting script."""
    cmd = ["python", plot.__file__, "--local_files", "--disable_tex"]
    run_verbose(cmd)
