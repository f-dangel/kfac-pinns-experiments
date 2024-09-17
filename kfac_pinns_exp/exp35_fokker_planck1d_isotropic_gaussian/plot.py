"""Plot the best runs from each tuned optimizer"""

from argparse import ArgumentParser
from itertools import product
from os import makedirs, path

from matplotlib import pyplot as plt
from palettable.colorbrewer import sequential
from tueplots import bundles

from kfac_pinns_exp.train import set_up_layers
from kfac_pinns_exp.wandb_utils import (
    WandbBayesianRunFormatter,
    WandbBayesianSweepFormatter,
    load_best_run,
    remove_unused_runs,
    show_sweeps,
)

entity = "kfac-pinns"  # team name on wandb
project = (
    "fokker_planck1d_isotropic_gaussian_bayes"  # name from the 'Projects' tab on wandb
)

# information for title
equation = "fokker-planck-isotropic"
architecture = "mlp-tanh-64"
dim_Omega = 1
num_params = sum(
    p.numel()
    for layer in set_up_layers(architecture, equation, dim_Omega)
    for p in layer.parameters()
)

# Useful to map sweep ids to human-readable names
print_sweeps = False
if print_sweeps:
    show_sweeps(entity, project)

sweep_ids = {  # ids from the wandb agent
    "agv2wzwh": "SGD",
    "83o2n0r5": "Adam",
    "u6stpt59": "Hessian-free",
    "8ddh1xq2": "LBFGS",
    "tm69ozk9": "ENGD (full)",
    "ty4vnipz": "ENGD (layer-wise)",
    "amx8wizu": "KFAC",
    "l43skq02": "KFAC*",
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

    x_to_xlabel = {"step": "Iteration", "time": "Time (s)"}
    y_to_ylabel = {"loss": "Loss", "l2_error": "$L_2$ error"}

    for (x, xlabel), (y, ylabel) in product(x_to_xlabel.items(), y_to_ylabel.items()):
        with plt.rc_context(
            bundles.neurips2023(rel_width=1.0, usetex=not args.disable_tex)
        ):
            fig, ax = plt.subplots(1, 1)
            ax.set_xlabel(xlabel)
            ax.set_xscale("log")
            ax.set_ylabel(ylabel)
            ax.set_yscale("log")
            ax.set_title(
                f"{dim_Omega}d {equation.capitalize()} ($D={num_params}$, Bayes)"
            )
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
                    label=label,
                    color=colors[label],
                    linestyle=linestyles[label],
                )

            ax.legend()
            plt.savefig(path.join(HEREDIR, f"{y}_over_{x}.pdf"), bbox_inches="tight")

    # export run descriptions to LaTeX
    TEXDIR = path.join(HEREDIR, "tex")
    makedirs(TEXDIR, exist_ok=True)

    if args.update:  # only if online access is possible
        for sweep_id in sweep_ids:
            _, meta = load_best_run(entity, project, sweep_id, savedir=DATADIR)
            sweep_args = meta.to_dict()["config"][0]
            WandbBayesianRunFormatter.to_tex(TEXDIR, sweep_args)

        for sweep in show_sweeps(entity, project):
            WandbBayesianSweepFormatter.to_tex(TEXDIR, sweep.config)
    else:
        print("Skipping LaTeX export of sweeps and best runs.")
