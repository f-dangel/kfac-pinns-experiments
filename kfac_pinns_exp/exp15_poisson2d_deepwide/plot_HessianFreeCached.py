"""Plot the best runs from each tuned optimizer, and the HessianFreeCached run."""

from argparse import ArgumentParser
from itertools import product
from os import path

from matplotlib import pyplot as plt
from tueplots import bundles

from kfac_pinns_exp.exp15_poisson2d_deepwide.plot import (
    DATADIR,
    HEREDIR,
    colors,
    dim_Omega,
    entity,
    equation,
    linestyles,
    num_params,
    project,
    sweep_ids,
)
from kfac_pinns_exp.wandb_utils import download_run, load_best_run

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

    y_to_ylabel = {
        # "loss": "Loss",
        "l2_error": "$L_2$ error",
    }
    x_to_xlabel = {
        "step": "Iteration",
        "time": "Time [s]",
    }

    for (x, xlabel), (y, ylabel) in product(x_to_xlabel.items(), y_to_ylabel.items()):
        with plt.rc_context(
            bundles.neurips2023(rel_width=0.5, usetex=not args.disable_tex)
        ):
            fig, ax = plt.subplots(1, 1)
            ax.set_xlabel(xlabel)
            ax.set_xscale("log")
            ax.set_ylabel(ylabel)
            ax.set_yscale("log")
            ax.set_title(f"{dim_Omega}d {equation.capitalize()} ($D={num_params}$)")
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

            # add HessianFreeCached run
            df_history, _ = download_run(
                entity,
                project,
                "HessianFreeCached-02",
                update=args.update,
                savedir=DATADIR,
            )
            label = "Hessian-free"
            x_data = {
                "step": df_history["step"] + 1,
                "time": df_history["time"] - min(df_history["time"]),
            }[x]
            ax.plot(
                x_data,
                df_history[y],
                label="Hessian-free + AD tricks",
                color=colors[label],
                linestyle=linestyles[label],
            )

            ax.legend()
            plt.savefig(
                path.join(HEREDIR, f"teaser_{y}_over_{x}.pdf"), bbox_inches="tight"
            )
