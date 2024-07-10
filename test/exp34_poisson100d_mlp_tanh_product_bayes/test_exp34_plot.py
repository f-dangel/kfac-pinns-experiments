"""Run the plotting script from exp34."""

from kfac_pinns_exp.exp34_poisson100d_mlp_tanh_product_bayes import plot
from kfac_pinns_exp.utils import run_verbose


def test_execute_plot():
    """Execute the plotting script."""
    cmd = ["python", plot.__file__, "--local_files", "--disable_tex"]
    run_verbose(cmd)
