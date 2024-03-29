"""Run the plotting script from exp10."""

from subprocess import CalledProcessError, run

from kfac_pinns_exp.exp10_reproduce_poisson5d import plot


def test_execute_plot():
    """Execute the plotting script.

    Raises:
        CalledProcessError: If the script fails.
    """
    cmd = ["python", plot.__file__, "--local_files", "--disable_tex"]

    try:
        run(cmd, capture_output=True, text=True, check=True)
    except CalledProcessError as e:
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise e
