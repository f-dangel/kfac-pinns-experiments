"""Create a pretty plot that groups together the results for 4d heat."""

from argparse import ArgumentParser
from itertools import product
from os import path

from matplotlib import pyplot as plt
from tueplots import bundles

from kfac_pinns_exp.exp27_heat4d_small.plot import (
    DATADIR,
    colors,
    entity,
    linestyles,
    num_params,
    project,
    sweep_ids,
)
from kfac_pinns_exp.exp28_heat4d_medium.plot import DATADIR as medium_DATADIR
from kfac_pinns_exp.exp28_heat4d_medium.plot import num_params as medium_num_params
from kfac_pinns_exp.exp28_heat4d_medium.plot import project as medium_project
from kfac_pinns_exp.exp28_heat4d_medium.plot import sweep_ids as medium_sweep_ids
from kfac_pinns_exp.exp29_heat4d_big.plot import DATADIR as BIG_DATADIR
from kfac_pinns_exp.exp29_heat4d_big.plot import num_params as big_num_params
from kfac_pinns_exp.exp29_heat4d_big.plot import project as big_project
from kfac_pinns_exp.exp29_heat4d_big.plot import sweep_ids as big_sweep_ids
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

    HEREDIR = path.dirname(path.abspath(__file__))
    IGNORE = {  # ignore the following optimizers for the plots in the main text
        "KFAC (empirical)",
        "KFAC (forward-only)",
        "KFAC* (empirical)",
        "KFAC* (forward-only)",
        "ENGD (diagonal)",
    }
    y_to_ylabel = {"loss": "Loss", "l2_error": "$L_2$ error"}
    APPENDIX = [False]  # show all optimizers in the appendix

    for appendix, (y, ylabel) in product(APPENDIX, y_to_ylabel.items()):
        # NOTE Use `nrows` and `ncols` to tweak the subplot size, because `tueplots`
        # always forces the plots length/height-ratio to be the golden ratio
        with plt.rc_context(
            bundles.neurips2023(
                rel_width=1.0, nrows=4, ncols=5, usetex=not args.disable_tex
            ),
        ):
            # update LaTeX preamble, so we can pretty-print the dimension titles
            plt.rcParams["text.latex.preamble"] = (
                plt.rcParams["text.latex.preamble"]
                + r"\usepackage[group-separator={,}, group-minimum-digits={3}]{siunitx}"
            )

            num_rows, num_cols = 2, 3
            fig, ax = plt.subplots(num_rows, num_cols)

            ax[0, 1].set_xlabel("Iteration")
            ax[1, 1].set_xlabel("Time [s]")
            ax[0, 0].set_ylabel(ylabel)
            ax[1, 0].set_ylabel(ylabel)

            for a in ax.flatten():
                a.grid(True, alpha=0.5)
                a.set_xscale("log")
                a.set_yscale("log")

            for col, row in product(range(num_cols), range(num_rows)):
                col_sweep_ids = [sweep_ids, medium_sweep_ids, big_sweep_ids][col]
                col_project = [project, medium_project, big_project][col]
                col_DATADIR = [DATADIR, medium_DATADIR, BIG_DATADIR][col]
                col_model_dims = [num_params, medium_num_params, big_num_params]
                col_title = [
                    f"$D={d}$" if args.disable_tex else r"$D=\num{" + str(d) + r"}$"
                    for d in col_model_dims
                ]

                for sweep_id, name in col_sweep_ids.items():
                    if not appendix and name in IGNORE:
                        continue

                    df_history, _ = load_best_run(
                        entity,
                        col_project,
                        sweep_id,
                        save=False,
                        # load data from other experiments to avoid duplication
                        update=False,
                        savedir=col_DATADIR,
                    )
                    x_data = {
                        0: df_history["step"] + 1,
                        1: df_history["time"] - min(df_history["time"]),
                    }[row]
                    label = name if (col == row == 0 and "*" not in name) else None
                    ax[row, col].plot(
                        x_data,
                        df_history[y],
                        label=label,
                        color=colors[name],
                        linestyle=linestyles[name],
                    )
                    title = col_title[col] if row == 0 else None
                    ax[row, col].set_title(title)

            # set x_min to 1 for time
            for a in ax[1, :]:
                a.set_xlim(left=1)

            handles, labels = ax[0, 0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="lower center",
                # adjust legend to not overlap with xlabel
                bbox_to_anchor=(0.5, -0.15 if appendix else -0.1),
                # shorter lines so legend fits into a single line in the main text
                handlelength=1.2,
                ncols=5 if appendix else 7,
            )
            suffix = "_all" if appendix else ""
            plt.savefig(path.join(HEREDIR, f"{y}{suffix}.pdf"), bbox_inches="tight")
