"""Run the profiling script for all cases."""

from itertools import product
from os import path

from torch import linspace

from kfac_pinns_exp.exp40_profile_laplacian.run import (
    DEFAULTDIR,
    IMPLEMENTATIONS,
    METRICS,
    get_raw_savepath,
)
from kfac_pinns_exp.utils import run_verbose

NUM_SEEDS = 5  # run multiple times
MODELS = ["mlp-tanh-768-768-512-512"]
DEVICES = ["cuda"]  # add 'cuda' here if you have a GPU
INPUT_DIMENSIONS = linspace(1, 100, 21).int().unique().tolist()
INPUT_DIMENSIONS = [1, 10, 100]
BATCH_SIZES = [1024]

if __name__ == "__main__":
    for (
        input_dimension,
        implementation,
        model,
        batch_size,
        device,
        metric,
        seed,
    ) in product(
        INPUT_DIMENSIONS,
        IMPLEMENTATIONS,
        MODELS,
        BATCH_SIZES,
        DEVICES,
        METRICS,
        range(NUM_SEEDS),
    ):
        savepath = get_raw_savepath(
            input_dimension,
            implementation,
            model,
            batch_size,
            device,
            seed,
            metric,
            DEFAULTDIR,
        )
        if path.exists(savepath):
            print(f"Already exists, skipping: {savepath}")
        else:
            cmd = [
                "python",
                "run.py",
                f"--input_dimension={input_dimension}",
                f"--implementation={implementation}",
                f"--batch_size={batch_size}",
                f"--model={model}",
                f"--device={device}",
                f"--metric={metric}",
                f"--seed={seed}",
            ]
            run_verbose(cmd)
