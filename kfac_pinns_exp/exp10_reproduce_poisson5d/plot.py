"""Plot the best runs from each tuned optimizer"""

from argparse import ArgumentParser
from os import makedirs, path

from matplotlib import pyplot as plt
from palettable.colorbrewer import sequential
from tueplots import bundles

from kfac_pinns_exp.exp09_reproduce_poisson2d.download_best import load_best_run

entity = "kfac-pinns"  # team name on wandb
project = "poisson5d"  # name from the 'Projects' tab on wandb

sweep_ids = {  # ids from the wandb agent
    "tu585kr5": "SGD",
    "5klsh2cl": "Adam",
    # "": "ENGD (full)",
    # "": "ENGD (layer-wise)",
    # "": "ENGD (diagonal)",
    # "": "Hessian-free",
}

# color options: https://jiffyclub.github.io/palettable/colorbrewer/
colors = {
    "SGD": sequential.Reds_4.mpl_colors[-2],
    "Adam": sequential.Reds_4.mpl_colors[-1],
    "ENGD (full)": sequential.Blues_5.mpl_colors[-3],
    "ENGD (layer-wise)": sequential.Blues_5.mpl_colors[-2],
    "ENGD (diagonal)": sequential.Blues_5.mpl_colors[-1],
    "Hessian-free": sequential.Greens_4.mpl_colors[-2],
    "KFAC": "black",
}

linestyles = {
    "SGD": "-",
    "Adam": "-",
    "ENGD (full)": "-",
    "ENGD (layer-wise)": "dashed",
    "ENGD (diagonal)": "dotted",
    "Hessian-free": "-",
    "KFAC": "-",
}

HEREDIR = path.dirname(path.abspath(__file__))
DATADIR = path.join(HEREDIR, "best_runs")
makedirs(DATADIR, exist_ok=True)

if __name__ == "__main__":
    parser = ArgumentParser(description="Plot the best runs from each tuned optimizer.")
    parser.add_argument(
        "--local_files",
        action="store_false",
        dest="update",
        help="Use local files if possible.",
        default=True,
    )
    parser.add_argument(
        "--disable_tex",
        action="store_true",
        default=False,
        help="Disable TeX rendering in matplotlib.",
    )
    args = parser.parse_args()

    with plt.rc_context(
        bundles.neurips2023(rel_width=1.0, usetex=not args.disable_tex)
    ):
        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel("Iteration")
        ax.set_xscale("log")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.set_title("Poisson 5d")

        for sweep_id, label in sweep_ids.items():
            df_history, _ = load_best_run(
                entity,
                project,
                sweep_id,
                save=True,
                update=args.update,
                savedir=DATADIR,
            )
            ax.plot(
                df_history["step"] + 1,
                df_history["loss"],
                label=label,
                color=colors[label],
                linestyle=linestyles[label],
            )

        ax.legend()

        plt.savefig(path.join(HEREDIR, "poisson5d.pdf"), bbox_inches="tight")
