"""Gather the data from individual runs."""

from itertools import product
from os import makedirs, path

from pandas import DataFrame

from kfac_pinns_exp.exp40_profile_laplacian.generate_data import (
    BATCH_SIZES,
    DEVICES,
    INPUT_DIMENSIONS,
    MODELS,
    NUM_SEEDS,
)
from kfac_pinns_exp.exp40_profile_laplacian.run import DEFAULTDIR as RAWDEFAULTDIR
from kfac_pinns_exp.exp40_profile_laplacian.run import (
    IMPLEMENTATIONS,
    METRICS,
    get_raw_savepath,
)

HEREDIR = path.dirname(path.abspath(__file__))
DEFAULTDIR = path.join(HEREDIR, "results", "gathered")
makedirs(DEFAULTDIR, exist_ok=True)


def get_gathered_savepath(
    implementation: str,
    model: str,
    batch_size: int,
    device: str,
    metric: str,
    datadir: str,
) -> str:
    """Get the save path for the gathered data.

    Args:
        implementation: The used implementation.
        model: The used model.
        batch_size: The used batch size.
        device: The used device.
        metric: The used metric.
        datadir: The used data directory.

    Returns:
        The save path for the gathered Laplacian data.
    """
    return path.join(
        datadir,
        "_".join([model, implementation, device, metric, f"N_{batch_size}"]) + ".csv",
    )


for implementation, model, batch_size, device, metric in product(
    IMPLEMENTATIONS, MODELS, BATCH_SIZES, DEVICES, METRICS
):
    best_metrics = []
    for input_dimension in INPUT_DIMENSIONS:
        # collect metric for all seeds, then use the best
        metrics = []
        for seed in range(NUM_SEEDS):
            savepath = get_raw_savepath(
                input_dimension,
                implementation,
                model,
                batch_size,
                device,
                seed,
                metric,
                RAWDEFAULTDIR,
            )
            with open(savepath, "r") as f:
                metrics.append(float(f.read().strip()))
        best_metrics.append(min(metrics))

    df = DataFrame.from_dict(
        {"input_dimension": INPUT_DIMENSIONS, metric: best_metrics}
    )
    savepath = get_gathered_savepath(
        implementation, model, batch_size, device, metric, DEFAULTDIR
    )
    df.to_csv(savepath, index=False)
