"""Create an error bar plot."""

from itertools import product
from os import path

from matplotlib import pyplot as plt
from tueplots import bundles

from kfac_pinns_exp.exp28_heat4d_medium.plot import (
    architecture,
    colors,
    linestyles,
    num_params,
)
from kfac_pinns_exp.exp41_errorbars_exp28.create_launch_script import (
    COMMANDS,
    REPEATDIR,
    entity,
    project,
    sweep_ids,
)
from kfac_pinns_exp.wandb_utils import download_run

HEREDIR = path.dirname(path.abspath(__file__))


y_to_ylabel = {"loss": "Loss", "l2_error": "$L_2$ error"}
x_to_xlabel = {"step": "Iteration", "time": "Time (s)"}

x = "step"
y = "loss"

for (x, xlabel), (y, ylabel) in product(x_to_xlabel.items(), y_to_ylabel.items()):
    with plt.rc_context(bundles.neurips2023(rel_width=1.0)):
        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel(xlabel)
        ax.set_xscale("log")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.set_title(f"4d Heat ({architecture}, $D={num_params}$)")
        ax.grid(True, alpha=0.5)

        from wandb.errors import CommError

        for sweep_id, best_runs in COMMANDS.items():
            for idx, run_id in enumerate(best_runs.keys()):
                if idx >= 1:
                    continue
                try:
                    df_history, _ = download_run(
                        entity,
                        project,
                        run_id,
                        savedir=REPEATDIR,
                        # update=False,
                    )
                    label = sweep_ids[sweep_id]

                    x_data = {
                        "step": df_history["step"] + 1,
                        "time": df_history["time"] - min(df_history["time"]),
                    }[x]

                    ax.plot(
                        x_data,
                        df_history[y],
                        label=None if "*" in label or idx != 0 else label,
                        color=colors[label],
                        linestyle=linestyles[label],
                    )
                except CommError:
                    pass

        # set y max to 10
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, min(ymax, 10))

        ax.legend()
        plt.savefig(path.join(HEREDIR, f"{y}_over_{x}.pdf"), bbox_inches="tight")
