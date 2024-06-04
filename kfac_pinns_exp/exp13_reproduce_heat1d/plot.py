"""Plot the best runs from each tuned optimizer"""

from argparse import ArgumentParser
from itertools import product
from os import makedirs, path

from matplotlib import pyplot as plt
from palettable.colorbrewer import sequential
from tueplots import bundles

from kfac_pinns_exp.train import set_up_layers
from kfac_pinns_exp.wandb_utils import (
    WandbRunFormatter,
    WandbSweepFormatter,
    load_best_run,
    remove_unused_runs,
    show_sweeps,
)

entity = "kfac-pinns"  # team name on wandb
project = "heat1d"  # name from the 'Projects' tab on wandb

architecture = "mlp-tanh-64"
num_params = sum(
    p.numel()
    for layer in set_up_layers(architecture, "heat", 1)
    for p in layer.parameters()
)

# Useful to map sweep ids to human-readable names
print_sweeps = False
if print_sweeps:
    show_sweeps(entity, project)

sweep_ids = {  # ids from the wandb agent
    "18jetn91": "SGD",
    "ows9kr21": "Adam",
    "ytlbsylk": "Hessian-free",
    "6734c6vy": "LBFGS",
    "ruxdkxho": "ENGD (full)",
    "0h3bytin": "ENGD (layer-wise)",
    "hfr4aook": "ENGD (diagonal)",
    "w8knpy8i": "KFAC",
    "sddr7a9a": "KFAC*",
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
    "KFAC*": "black",
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
    "KFAC*": "dashed",
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

    y_to_ylabel = {"loss": "Loss", "l2_error": "$L_2$ error"}
    x_to_xlabel = {"step": "Iteration", "time": "Time (s)"}

    for (x, xlabel), (y, ylabel) in product(x_to_xlabel.items(), y_to_ylabel.items()):
        with plt.rc_context(
            bundles.neurips2023(rel_width=1.0, usetex=not args.disable_tex)
        ):
            fig, ax = plt.subplots(1, 1)
            ax.set_xlabel(xlabel)
            ax.set_xscale("log")
            ax.set_ylabel(ylabel)
            ax.set_yscale("log")
            ax.set_title(f"1d Heat ({architecture}, $D={num_params}$)")
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
                x_data = {
                    "step": df_history["step"] + 1,
                    "time": df_history["time"] - min(df_history["time"]),
                }[x]
                ax.plot(
                    x_data,
                    df_history[y],
                    label=None if "*" in label else label,
                    color=colors[label],
                    linestyle=linestyles[label],
                )

            ax.legend()
            plt.savefig(path.join(HEREDIR, f"{y}_over_{x}.pdf"), bbox_inches="tight")

    # export sweep and run descriptions to LaTeX
    TEXDIR = path.join(HEREDIR, "tex")
    makedirs(TEXDIR, exist_ok=True)

    if args.update:  # only if online access is possible
        for sweep_id in sweep_ids:
            _, meta = load_best_run(entity, project, sweep_id, savedir=DATADIR)
            sweep_args = meta.to_dict()["config"][0]
            WandbRunFormatter.to_tex(TEXDIR, sweep_args)

        for sweep in show_sweeps(entity, project):
            WandbSweepFormatter.to_tex(TEXDIR, sweep.config)
    else:
        print("Skipping LaTeX export of sweeps and best runs.")
