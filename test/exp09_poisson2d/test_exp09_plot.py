"""Run the plotting script from exp09."""

from subprocess import CalledProcessError, run

from kfac_pinns_exp.exp09_reproduce_poisson2d import plot


def test_execute_plot():
    """Execute the plotting script."""
    cmd = ["python", plot.__file__, "--local_files", "--disable_tex"]

    try:
        run(cmd, capture_output=True, text=True, check=True)
    except CalledProcessError as e:
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise e
