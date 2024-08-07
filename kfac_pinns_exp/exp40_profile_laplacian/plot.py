"""Plot the gathered data."""

from itertools import product
from os import makedirs, path

from matplotlib import pyplot as plt
from pandas import read_csv
from tueplots import bundles

from kfac_pinns_exp.exp40_profile_laplacian.gather_data import (
    DEFAULTDIR as GATHERDEFAULTDIR,
)
from kfac_pinns_exp.exp40_profile_laplacian.gather_data import get_gathered_savepath
from kfac_pinns_exp.exp40_profile_laplacian.generate_data import (
    BATCH_SIZES,
    DEVICES,
    MODELS,
)
from kfac_pinns_exp.exp40_profile_laplacian.run import IMPLEMENTATIONS, METRICS

HEREDIR = path.dirname(path.abspath(__file__))
DEFAULTDIR = path.join(HEREDIR, "fig")
makedirs(DEFAULTDIR, exist_ok=True)


def get_plot_savepath(
    model: str,
    batch_size: int,
    device: str,
    metric: str,
    datadir: str,
) -> str:
    """Get the save path for the plot.

    Args:
        model: The used model.
        batch_size: The used batch size.
        device: The used device.
        metric: The used metric.
        datadir: The used directory.

    Returns:
        The save path for the plot.
    """
    return path.join(
        datadir,
        "_".join([model, device, metric, f"N_{batch_size}"]) + ".pdf",
    )


LEGENDENTRIES = {
    "backward": r"\texttt{functorch} Laplacian",
    "forward": r"Forward Laplacian",
}
YLABELS = {
    "time": "Time [s]",
    "peakmem": "Peak memory [GiB]",
}
MARKERSTYLES = {
    "backward": "o",
    "forward": "s",
}

if __name__ == "__main__":
    for model, batch_size, device, metric in product(
        MODELS, BATCH_SIZES, DEVICES, METRICS
    ):
        savepath = get_plot_savepath(model, batch_size, device, metric, DEFAULTDIR)

        with plt.rc_context(bundles.neurips2023(rel_width=0.5)):
            fig, ax = plt.subplots(1, 1)
            ax.set_xlabel(r"$D_\Omega$")
            ax.set_ylabel(YLABELS[metric])

            for implementation in IMPLEMENTATIONS:
                gathered_savepath = get_gathered_savepath(
                    implementation, model, batch_size, device, metric, GATHERDEFAULTDIR
                )
                df = read_csv(gathered_savepath)
                ax.plot(
                    df["input_dimension"],
                    df[metric],
                    label=LEGENDENTRIES[implementation],
                    marker=MARKERSTYLES[implementation],
                    markersize=3,
                )

            ax.legend()
            plt.savefig(savepath, bbox_inches="tight")
            plt.close()
