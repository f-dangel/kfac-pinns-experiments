"""Create a pretty plot that groups together the results for 4d heat."""

from argparse import ArgumentParser
from itertools import product
from os import path

from matplotlib import pyplot as plt
from tueplots import bundles

from kfac_pinns_exp.exp27_heat4d_small import plot as SMALL
from kfac_pinns_exp.exp27_heat4d_small.plot import colors, linestyles
from kfac_pinns_exp.exp28_heat4d_medium import plot as MEDIUM
from kfac_pinns_exp.exp29_heat4d_big import plot as BIG
from kfac_pinns_exp.wandb_utils import load_best_run

if __name__ == "__main__":
    parser = ArgumentParser(description="Summarize the experiments on heat 4d")
    parser.add_argument(
        "--disable_tex",
        action="store_true",
        default=False,
        help="Disable TeX rendering in matplotlib.",
    )
    args = parser.parse_args()

    COLUMNS = [SMALL, MEDIUM, BIG]  # which sub-experiment to plot in which column
    IGNORE = {"ENGD (diagonal)"}

    # Create plots of all combinations of x and y
    y_to_ylabel = {"loss": "Loss", "l2_error": "$L_2$ error"}
    x_to_xlabel = {"step": "Iteration", "time": "Time [s]"}

    for (x, xlabel), (y, ylabel) in product(x_to_xlabel.items(), y_to_ylabel.items()):
        # NOTE Use `nrows` and `ncols` to tweak the subplot size, because `tueplots`
        # always forces the plots length/height-ratio to be the golden ratio
        with plt.rc_context(
            bundles.neurips2023(
                rel_width=1.0,
                nrows=4,
                ncols=3 * len(COLUMNS),
                usetex=not args.disable_tex,
            ),
        ):
            # update LaTeX preamble, so we can pretty-print the dimension titles
            plt.rcParams["text.latex.preamble"] = (
                plt.rcParams["text.latex.preamble"]
                + r"\usepackage[group-separator={,}, group-minimum-digits={3}]{siunitx}"
            )

            num_rows, num_cols = 1, len(COLUMNS)
            fig, axs = plt.subplots(num_rows, num_cols)

            axs[0].set_ylabel(ylabel)
            for ax in axs.flatten():
                ax.set_xlabel(xlabel)
                ax.grid(True, alpha=0.5)
                ax.set_xscale("log")
                ax.set_yscale("log")

            for ax, exp in zip(axs, COLUMNS):
                D = exp.num_params
                title = f"$D={D}$" if args.disable_tex else r"$D=\num{" + str(D) + r"}$"
                ax.set_title(title)

                for sweep_id, name in exp.sweep_ids.items():
                    if name in IGNORE:
                        continue

                    df_history, _ = load_best_run(
                        exp.entity,
                        exp.project,
                        sweep_id,
                        save=False,
                        # load data from other experiments to avoid duplication
                        update=False,
                        savedir=exp.DATADIR,
                    )
                    x_data = {
                        "step": df_history["step"] + 1,
                        "time": df_history["time"] - min(df_history["time"]),
                    }[x]
                    label = name if ax is axs[0] else None
                    ax.plot(
                        x_data,
                        df_history[y],
                        label=label,
                        color=colors[name],
                        linestyle=linestyles[name],
                    )

            # set x_min to 1 for time
            if x == "time":
                for ax in axs.flatten():
                    ax.set_xlim(left=1)

            handles, labels = axs[0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="lower center",
                # adjust legend to not overlap with xlabel
                bbox_to_anchor=(0.5, -0.13),
                # shorter lines so legend fits into a single line in the main text
                handlelength=1.35,
                # reduce space between columns to fit into a single line
                ncols=8,
                columnspacing=0.9,
            )
            HEREDIR = path.dirname(path.abspath(__file__))
            plt.savefig(path.join(HEREDIR, f"{y}_over_{x}.pdf"), bbox_inches="tight")
