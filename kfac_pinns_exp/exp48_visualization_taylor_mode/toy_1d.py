"""Plot functions and Taylor expansions of a one-dimensional function composition."""

from typing import Callable

from matplotlib import pyplot as plt
from torch import Tensor, linspace, manual_seed
from torch.autograd.functional import hessian, jacobian
from torch.nn import Identity, Module, Sequential, Tanh


class ScaledTanh(Tanh):
    def forward(self, x: Tensor) -> Tensor:
        return 2 * super().forward(x)


class Cubic(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x**3


class Sin(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sin()


def taylor(
    f: Callable[[Tensor], Tensor], x0: Tensor, xs: Tensor, degree: int
) -> Tensor:
    """Compute the Taylor expansion of a 1d function around a point.

    Args:
        f: The function to be expanded.
        x0: The point around which to expand.
        xs: The values for which to evaluate the Taylor expansion.
        degree: The degree of the Taylor expansion.

    Raises:
        NotImplementedError: If the degree is not in {0, 1, 2}.
        NotImplementedError: If the point x0 is not a 1d tensor.
        NotImplementedError: If the values xs are not a 1d tensor.

    Returns:
        The Taylor expansion of the function f around the point x0.
    """
    if degree not in {0, 1, 2}:
        raise NotImplementedError("The degree must be in {0, 1, 2}.")
    if x0.shape != (1,):
        raise NotImplementedError("The point x0 must be a 1d tensor.")
    if xs.ndim != 1:
        raise NotImplementedError("The values xs must be a 1d tensor.")

    f_taylor = f(x0).expand_as(xs)
    jac = jacobian(f, x0).item()
    hess = hessian(f, x0).item()

    shift = xs - x0.item()

    if degree >= 1:
        f_taylor = f_taylor + jac * shift
    if degree >= 2:
        f_taylor = f_taylor + 0.5 * shift * hess * shift

    return f_taylor


if __name__ == "__main__":
    manual_seed(0)

    # functions to be composed
    layers = [Identity(), ScaledTanh(), Cubic(), Sin()]

    # values for which to visualize the function and its Taylor expansions
    degrees = [0, 1, 2]
    x = Tensor([0.5])
    xs = linspace(x.item() - 0.75, x.item() + 0.75, 150)
    xs_taylor = linspace(x.item() - 0.35, x.item() + 0.35, 100)

    # styles for plotting
    style = {"linewidth": 8.5, "color": "blue"}
    taylor_style = {"linestyle": "dashed", "linewidth": 8.5, "color": "orange"}
    marker_style = {"marker": "o", "markersize": 18, "color": "orange"}

    # visualize the function
    for i in range(len(layers)):
        # compute the function and its Taylor expansions
        f = Sequential(*layers[: i + 1])
        f_x = f(x)
        ys = f(xs.unsqueeze(1)).squeeze(1)
        ys_taylor = {d: taylor(f, x, xs_taylor, d) for d in degrees}

        for degree in [None] + degrees:
            fig, ax = plt.subplots()
            ax.axis("off")
            ax.plot(xs, ys, **style)

            # invisibly plot the other Taylor expansions so we get the same range
            [
                ax.plot(xs_taylor, ys_taylor, **taylor_style, alpha=0)
                for ys_taylor in ys_taylor.values()
            ]

            # plot the Taylor expansion
            if degree is not None:
                ax.plot(xs_taylor, ys_taylor[degree], **taylor_style)
                ax.plot(x, f_x, **marker_style)

            # save the plot
            savepath = (
                f"f_{i}" + ("" if degree is None else f"_taylor_{degree}") + ".pdf"
            )
            plt.savefig(savepath, bbox_inches="tight", transparent=True)
            plt.close()
