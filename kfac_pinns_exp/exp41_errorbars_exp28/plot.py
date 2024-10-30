"""Create an error bar plot."""

from argparse import ArgumentParser
from itertools import product
from os import path

from matplotlib import pyplot as plt
from numpy import mean, std
from tueplots import bundles

from kfac_pinns_exp.exp28_heat4d_medium.plot import (
    architecture,
    colors,
    linestyles,
    num_params,
)
from kfac_pinns_exp.exp41_errorbars_exp28.create_launch_script import (
    REPEATDIR,
    entity,
    get_commands,
    project,
    sweep_ids,
)
from kfac_pinns_exp.wandb_utils import download_run

if __name__ == "__main__":
    parser = ArgumentParser(description="Create errorbar plots.")
    parser.add_argument(
        "--local_files",
        action="store_true",
        dest="local_files",
        help="Use local files if possible.",
        default=False,
    )
    parser.add_argument(
        "--disable_tex",
        action="store_true",
        default=False,
        help="Disable TeX rendering in matplotlib.",
    )
    args = parser.parse_args()

    COMMANDS = get_commands(local_files=args.local_files)
    HEREDIR = path.dirname(path.abspath(__file__))

    y_to_ylabel = {"loss": "Loss", "l2_error": "$L_2$ error"}
    x_to_xlabel = {"step": "Iteration", "time": "Time [s]"}

    final_performance = {
        "loss": {sweep_ids[sweep_id]: [] for sweep_id in COMMANDS.keys()},
        "l2_error": {sweep_ids[sweep_id]: [] for sweep_id in COMMANDS.keys()},
    }

    for plot_idx, ((x, xlabel), (y, ylabel)) in enumerate(
        product(x_to_xlabel.items(), y_to_ylabel.items())
    ):
        with plt.rc_context(
            bundles.neurips2023(rel_width=1.0, usetex=not args.disable_tex)
        ):
            fig, ax = plt.subplots(1, 1)
            ax.set_xlabel(xlabel)
            ax.set_xscale("log")
            ax.set_ylabel(ylabel)
            ax.set_yscale("log")
            ax.set_title(f"4d Heat ({architecture}, $D={num_params}$)")
            ax.grid(True, alpha=0.5)

            for sweep_id, best_runs in COMMANDS.items():
                for idx, run_id in enumerate(best_runs.keys()):
                    df_history, _ = download_run(
                        entity,
                        project,
                        run_id,
                        savedir=REPEATDIR,
                        # only download data again for first sub-plot, then re-use
                        update=plot_idx == 0 and not args.local_files,
                    )
                    label = sweep_ids[sweep_id]

                    x_data = {
                        "step": df_history["step"] + 1,
                        "time": df_history["time"] - min(df_history["time"]),
                    }[x]

                    ax.plot(
                        x_data,
                        df_history[y],
                        label=label if idx == 0 else None,
                        color=colors[label],
                        linestyle=linestyles[label],
                    )
                    final_performance[y][label].append(df_history[y].tolist()[-1])

            if x == "time" and y == "l2_error":
                ax.legend()

            # set y max to 10 because LBFGS sometimes diverges
            # also need to set the bottom value because otherwise it is way too small
            bottom = {"loss": 1e-11, "l2_error": 5e-6}[y]
            plt.gca().set_ylim(bottom=bottom, top=1e1)

            plt.savefig(path.join(HEREDIR, f"{y}_over_{x}.pdf"), bbox_inches="tight")

    # print(final_performance)
    # print final performance
    for metric, optim_performance in final_performance.items():
        print(f"{metric}:")
        for name, performance in optim_performance.items():
            print(f"\t{name:<20}: {mean(performance):.2e} ± {std(performance):.2e}")
