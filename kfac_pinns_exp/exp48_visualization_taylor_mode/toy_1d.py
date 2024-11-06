"""Plot functions and Taylor expansions of a one-dimensional function composition."""

from math import factorial
from os import makedirs, path
from typing import Callable

from matplotlib import pyplot as plt
from torch import Tensor, linspace, vmap, zeros_like
from torch.func import grad
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
        x0: The point around which to expand (scalar tensor).
        xs: The values for which to evaluate the Taylor expansion.
        degree: The degree of the Taylor expansion.

    Raises:
        NotImplementedError: If the point x0 is not a scalar tensor.
        NotImplementedError: If the values xs are not a 1d tensor.

    Returns:
        The Taylor expansion of the function f around the point x0.
        Has same shape as `xs`.
    """
    if x0.shape != ():
        raise NotImplementedError("The point x0 must be a 0d tensor.")
    if xs.ndim != 1:
        raise NotImplementedError("The values xs must be a 1d tensor.")

    @vmap
    def f_taylor(x: Tensor) -> Tensor:
        """Evaluate the Taylor expansion of f around x0 at x.

        Args:
            x: The value for which to evaluate the Taylor expansion (scalar tensor).

        Returns:
            The value of the Taylor expansion at x.
        """
        dk_f = f
        f_x = zeros_like(x)

        for k in range(degree + 1):
            f_x += dk_f(x0) * (x - x0) ** k / factorial(k)
            dk_f = grad(dk_f)

        return f_x

    return f_taylor(xs)


if __name__ == "__main__":
    FIGDIR = path.join(path.dirname(path.abspath(__file__)), "figures")
    makedirs(FIGDIR, exist_ok=True)

    # functions to be composed
    layers = [Identity(), ScaledTanh(), Cubic(), Sin()]

    # values for which to visualize the function and its Taylor expansions
    max_degree = 6
    degrees = list(range(max_degree + 1))
    x = Tensor([0.5]).squeeze()
    xs = linspace(x.item() - 1.2, x.item() + 1.2, 150)
    xs_taylor = linspace(x.item() - 0.55, x.item() + 0.55, 100)

    # styles for plotting
    style = {"linewidth": 8.5, "color": "blue"}
    densely_dashed = (
        0,
        (5, 1),
    )
    taylor_style = {"linestyle": densely_dashed, "linewidth": 8.5, "color": "orange"}
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
            savepath = path.join(FIGDIR, savepath)
            plt.savefig(savepath, bbox_inches="tight", transparent=True)
            plt.close()
