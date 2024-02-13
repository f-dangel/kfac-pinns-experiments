"""Execute the training script in `exp09` (integration test)."""

from subprocess import check_output

from kfac_pinns_exp.exp09_kfac_optimizer import train


def test_run_exp09_train():
    """Execute the training script."""
    cmd = ["python", train.__file__]
    check_output(cmd)
