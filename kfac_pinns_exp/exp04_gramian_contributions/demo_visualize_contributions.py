"""Visualize the different contributions to the Gram matrix on a toy problem."""

from itertools import product
from os import makedirs, path

import matplotlib.pyplot as plt
from torch import allclose, manual_seed, rand, zeros_like
from torch.nn import Linear, Sequential, Sigmoid

from kfac_pinns_exp.autodiff_utils import autograd_gramian
from kfac_pinns_exp.exp04_gramian_contributions.demo_gramian_contributions import (
    CHILDREN,
    get_block_idx,
    get_layer_idx_and_name,
    gramian_term,
)
from kfac_pinns_exp.utils import combine_tiles, separate_into_tiles

HEREDIR = path.dirname(path.abspath(__file__))
FIGDIR = path.join(HEREDIR, "fig")
makedirs(FIGDIR, exist_ok=True)


def main():
    """Visualize the different contributions to the Gram matrix on a toy problem."""
    # setup
    manual_seed(0)
    batch_size = 10
    X = rand(batch_size, 5)
    layers = [
        Linear(5, 4),
        Sigmoid(),
        Linear(4, 3),
        Sigmoid(),
        Linear(3, 2),
        Sigmoid(),
        Linear(2, 1),
    ]
    model = Sequential(*layers)

    gram = autograd_gramian(model, X, [name for name, _ in model.named_parameters()])

    # compute the contributions to the full Gramian in tiles
    dims = [p.numel() for p in model.parameters()]
    contributions = {
        (child1, child2): separate_into_tiles(zeros_like(gram), dims)
        for child1, child2 in product(CHILDREN, CHILDREN)
    }

    for param1, param2, child1, child2 in product(
        model.parameters(), model.parameters(), CHILDREN, CHILDREN
    ):
        layer_idx1, param_name1 = get_layer_idx_and_name(param1, layers)
        layer_idx2, param_name2 = get_layer_idx_and_name(param2, layers)
        block_idx1 = get_block_idx(param1, model)
        block_idx2 = get_block_idx(param2, model)

        contributions[(child1, child2)][block_idx1][block_idx2].add_(
            gramian_term(
                layers,
                X,
                layer_idx1,
                param_name1,
                child1,
                layer_idx2,
                param_name2,
                child2,
                flat_params=True,
            )
        )

    # combine tiles into matrices
    contributions = {
        key: combine_tiles(values) for key, values in contributions.items()
    }

    # make sure the contributions sum up to the full Gramian
    assert allclose(sum(contributions.values()), gram)

    # visualize the contributions, use a shared color limit

    vmin = min(*[c.min() for c in contributions.values()], gram.min())
    vmax = max(*[c.max() for c in contributions.values()], gram.max())

    # visualize the Gram matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")  # turn off ticks and tick labels
    im = ax.imshow(gram, vmin=vmin, vmax=vmax)
    # positioning of the color bar from https://stackoverflow.com/a/43425119
    fig.colorbar(im, orientation="horizontal", pad=0.1)
    fig.savefig(path.join(FIGDIR, "gram_full.png"), bbox_inches="tight")
    plt.close(fig)

    # visualize the contributions
    done = []
    for child1, child2 in contributions.keys():
        if {child1, child2} in done:
            continue

        mat = contributions[(child1, child2)]
        if child1 != child2:
            mat = mat + contributions[(child2, child1)]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis("off")  # turn off ticks and tick labels
        im = ax.imshow(mat, vmin=vmin, vmax=vmax)
        plt.savefig(
            path.join(FIGDIR, f"gram_{child1}_{child2}.png"), bbox_inches="tight"
        )
        plt.close(fig)
        done.append({child1, child2})


if __name__ == "__main__":
    main()
