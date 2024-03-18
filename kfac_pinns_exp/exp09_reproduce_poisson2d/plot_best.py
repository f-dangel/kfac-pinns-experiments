"""Plot the best runs from each tuned optimizer"""

from matplotlib import pyplot as plt
from tueplots import bundles

from kfac_pinns_exp.exp09_reproduce_poisson2d.download_best import load_best_run

entity = "kfac-pinns"  # team name on wandb
project = "kfac-pinns-kfac_pinns_exp"  # name from the 'Projects' tab on wandb

sweep_ids = {  # ids from the wandb agent
    "gt86q21q": "GD",
    "1sie9nvg": "Adam",
    # TODO KFAC
    # TODO ENGD
    # TODO ENGD block-diagonal
    # TODO ENGD diagonal
    # TODO HessianFree
}

if __name__ == "__main__":
    with plt.rc_context(bundles.neurips2023(rel_width=0.5)):
        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.set_title("Poisson 2d")

        for sweep_id, label in sweep_ids.items():
            df_history, _ = load_best_run(
                entity, project, sweep_id, save=True, update=True
            )
            ax.plot(df_history["step"], df_history["loss"], label=label)

        ax.legend()

        plt.savefig("poisson2d.pdf", bbox_inches="tight")
