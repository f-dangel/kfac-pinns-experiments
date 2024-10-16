"""Plots solutions of checkpoints."""

from argparse import ArgumentParser
from glob import glob
from os import makedirs, path

from torch import load
from torch.nn import Sequential

from kfac_pinns_exp import (
    fokker_planck_isotropic_equation,
    heat_equation,
    log_fokker_planck_isotropic_equation,
    poisson_equation,
)
from kfac_pinns_exp.pinn_utils import l2_error
from kfac_pinns_exp.train import SOLUTIONS, set_up_layers
from kfac_pinns_exp.utils import latex_float

PLOT_FNS = {
    "poisson": poisson_equation.plot_solution,
    "heat": heat_equation.plot_solution,
    "fokker-planck-isotropic": fokker_planck_isotropic_equation.plot_solution,
    "log-fokker-planck-isotropic": log_fokker_planck_isotropic_equation.plot_solution,
}


def visualize_checkpoint(checkpoint: str, plot_dir: str, disable_tex: bool):
    """Generate a plot of the true and learned solution for a checkpoint.

    Args:
        checkpoint: The path to the checkpoint.
        plot_dir: The directory where the plot should be saved.
        disable_tex: Whether to disable LaTeX rendering in the plot.
    """
    print(f"Visualizing checkpoint {checkpoint}.")
    data = load(checkpoint)

    X_Omega_eval = data["X_Omega_eval"]
    step = data["step"]
    loss = data["loss"]

    config = data["config"]
    equation = config["equation"]
    dim_Omega = config["dim_Omega"]
    condition = config["boundary_condition"]
    architecture = config["model"]

    model = Sequential(*set_up_layers(architecture, equation, dim_Omega)).to(
        X_Omega_eval.dtype
    )
    model.load_state_dict(data["model"])

    u = SOLUTIONS[equation][condition]
    l2 = l2_error(model, X_Omega_eval, u)

    fig_path = path.join(
        plot_dir, f"{path.basename(checkpoint).replace('.pt', '.pdf')}"
    )
    print(f"Saving plot to {fig_path}.")
    fig_title = (
        f"Step: ${step}$, Loss: ${latex_float(loss)}$"
        + f" $L_2$ loss: ${latex_float(l2.item())}$"
    )

    plot_fn = PLOT_FNS[equation]
    plot_fn(
        condition, dim_Omega, model, fig_path, title=fig_title, usetex=not disable_tex
    )


def main():
    """Visualize the solution for each checkpoint."""
    parser = ArgumentParser(description="Plot solutions of checkpoints.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory containing checkpoints that should be visualized.",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="visualize_solution",
        help="Directory where the plots should be saved.",
    )
    parser.add_argument(
        "--disable_tex",
        action="store_true",
        help="Disable LaTeX rendering in plots.",
    )
    args = parser.parse_args()

    makedirs(args.plot_dir, exist_ok=True)

    checkpoint_dir = path.abspath(args.checkpoint_dir)
    plot_dir = path.abspath(args.plot_dir)

    for checkpoint in glob(path.join(checkpoint_dir, "*.pt")):
        visualize_checkpoint(checkpoint, plot_dir, args.disable_tex)


if __name__ == "__main__":
    main()
