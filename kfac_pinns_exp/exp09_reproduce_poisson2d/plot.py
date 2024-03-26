"""Plot the best runs from each tuned optimizer"""

from matplotlib import pyplot as plt
from palettable.colorbrewer import sequential
from tueplots import bundles

from kfac_pinns_exp.exp09_reproduce_poisson2d.download_best import load_best_run

entity = "kfac-pinns"  # team name on wandb
project = "poisson2d"  # name from the 'Projects' tab on wandb

sweep_ids = {  # ids from the wandb agent
    "tg2odbah": "SGD",
    "yzdbc4h3": "Adam",
    "as97g04v": "ENGD (full)",
    "svlfa1az": "ENGD (layer-wise)",
    "vk7egia5": "ENGD (diagonal)",
    "lmzgj39h": "Hessian-free",
    # TODO KFAC
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

if __name__ == "__main__":
    with plt.rc_context(bundles.neurips2023(rel_width=1.0)):
        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel("Iteration")
        ax.set_xscale("log")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.set_title("Poisson 2d")

        for sweep_id, label in sweep_ids.items():
            df_history, _ = load_best_run(
                entity, project, sweep_id, save=True, update=True
            )
            ax.plot(
                df_history["step"] + 1,
                df_history["loss"],
                label=label,
                color=colors[label],
                linestyle=linestyles[label],
            )

        ax.legend()
        plt.savefig("poisson2d.pdf", bbox_inches="tight")
