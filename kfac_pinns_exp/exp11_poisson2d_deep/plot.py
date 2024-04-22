"""Plot the best runs from each tuned optimizer"""

from argparse import ArgumentParser
from os import makedirs, path

from matplotlib import pyplot as plt
from palettable.colorbrewer import sequential
from tueplots import bundles

from kfac_pinns_exp.wandb_utils import load_best_run, remove_unused_runs, show_sweeps

entity = "kfac-pinns"  # team name on wandb
project = "poisson2d_deep"  # name from the 'Projects' tab on wandb

# Useful to map sweep ids to human-readable names
print_sweeps = False
if print_sweeps:
    show_sweeps(entity, project)

sweep_ids = {  # ids from the wandb agent
    "lbbpybzv": "SGD",
    "3fyoofbh": "Adam",
    "4nml4taz": "Hessian-free",
    "jytjr3ih": "LBFGS",
    "4eikh376": "ENGD (full)",
    "in7aq9o4": "ENGD (layer-wise)",
    "tosqfoxn": "ENGD (diagonal)",
    "9ro87rxo": "KFAC",
    # "qwied11a": "KFAC (empirical)",
    # "67z4wzfk": "KFAC (forward-only)",
}

# color options: https://jiffyclub.github.io/palettable/colorbrewer/
colors = {
    "SGD": sequential.Reds_4.mpl_colors[-2],
    "Adam": sequential.Reds_4.mpl_colors[-1],
    "ENGD (full)": sequential.Blues_5.mpl_colors[-3],
    "ENGD (layer-wise)": sequential.Blues_5.mpl_colors[-2],
    "ENGD (diagonal)": sequential.Blues_5.mpl_colors[-1],
    "Hessian-free": sequential.Greens_4.mpl_colors[-2],
    "LBFGS": sequential.Greens_4.mpl_colors[-1],
    "KFAC": "black",
    "KFAC (empirical)": "gray",
    "KFAC (forward-only)": "lightgray",
}

linestyles = {
    "SGD": "-",
    "Adam": "-",
    "ENGD (full)": "-",
    "ENGD (layer-wise)": "-",
    "ENGD (diagonal)": "-",
    "Hessian-free": "-",
    "LBFGS": "-",
    "KFAC": "-",
    "KFAC (empirical)": "-",
    "KFAC (forward-only)": "-",
}

HEREDIR = path.dirname(path.abspath(__file__))
DATADIR = path.join(HEREDIR, "best_runs")
makedirs(DATADIR, exist_ok=True)

# enable this to remove all saved files from sweeps that are not plotted
clean_up = True
if clean_up:
    remove_unused_runs(keep=list(sweep_ids.keys()), best_run_dir=DATADIR)

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

    metric_to_ylabel = {"loss": "Loss", "l2_error": "$L_2$ error"}

    for metric, ylabel in metric_to_ylabel.items():
        with plt.rc_context(
            bundles.neurips2023(rel_width=1.0, usetex=not args.disable_tex)
        ):
            fig, ax = plt.subplots(1, 1)
            ax.set_xlabel("Iteration")
            ax.set_xscale("log")
            ax.set_ylabel(ylabel)
            ax.set_yscale("log")
            ax.set_title("2d Poisson (deep net)")
            ax.grid(True, alpha=0.5)

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
                    df_history[metric],
                    label=label,
                    color=colors[label],
                    linestyle=linestyles[label],
                )

            ax.legend()
            plt.savefig(
                path.join(HEREDIR, f"poisson2d_deep_{metric}.pdf"), bbox_inches="tight"
            )
