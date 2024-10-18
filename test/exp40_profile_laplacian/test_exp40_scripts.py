"""Execute the scripts in `exp42`."""

from typing import List

from pytest import mark

from kfac_pinns_exp.exp40_profile_laplacian import run
from kfac_pinns_exp.utils import run_verbose

IMPLEMENTATIONS = ["backward", "forward"]
METRICS = ["time", "peakmem"]
COMMANDS = [
    [
        "--input_dimension=3",
        "--batch_size=32",
        "--model=mlp-tanh-64-48-32-16",
        "--device=cpu",
        "--seed=4",
    ],
]
COMMAND_IDS = ["_".join(cmd) for cmd in COMMANDS]


@mark.parametrize("implementation", IMPLEMENTATIONS, ids=IMPLEMENTATIONS)
@mark.parametrize("metric", METRICS, ids=METRICS)
@mark.parametrize("cmd", COMMANDS, ids=COMMAND_IDS)
def test_benchmarking_script(cmd: List[str], implementation: str, metric: str):
    """Execute the benchmarking script.

    Args:
        cmd: Command line arguments.
        implementation: Implementation to profile.
        metric: Metric to profile.
    """
    cmd = [
        "python",
        run.__file__,
        *cmd,
        f"--implementation={implementation}",
        f"--metric={metric}",
    ]
    run_verbose(cmd)
