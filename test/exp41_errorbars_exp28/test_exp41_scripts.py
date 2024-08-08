"""Execute the scripts in `exp41`."""

from kfac_pinns_exp.exp41_errorbars_exp28 import create_launch_script, plot
from kfac_pinns_exp.utils import run_verbose


def test_create_launch_script():
    """Execute the launch script generator script."""
    cmd = ["python", create_launch_script.__file__, "--local_files"]
    run_verbose(cmd)


def test_plotting_script():
    """Execute the plotting script."""
    cmd = ["python", plot.__file__, "--local_files", "--disable_tex"]
    run_verbose(cmd)
