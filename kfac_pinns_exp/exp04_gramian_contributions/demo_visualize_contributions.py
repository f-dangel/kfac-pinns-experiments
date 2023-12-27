"""Visualize the different contributions to the Gram matrix on a toy problem."""

from itertools import product
from os import makedirs, path
from typing import List

import matplotlib.pyplot as plt
from numpy import cumsum
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


def highlight_block_diagonal(ax: plt.Axes, dims: List[int]):
    """Highlight block diagonals.

    Args:
        ax: Axes onto which the highlighting will be applied.
        dims: Dimensions of the blocks on the diagonal.
    """
    style = {
        "linewidth": 1,
        "color": "w",
    }

    boundaries = cumsum([0] + dims)
    # need to provide relative values in [0; 1] for line lengths
    boundaries_rel = boundaries / boundaries[-1]

    for i in range(len(boundaries) - 1):
        start, end = boundaries[i : i + 2]
        start_rel, end_rel = boundaries_rel[i : i + 2]

        # NOTE imshow centers pixels at integers -> shift by -0.5
        # NOTE y axis orientation for imshow is inverse of that for axvline -> 1-...
        ax.axhline(y=start - 0.5, xmin=start_rel, xmax=end_rel, **style)
        ax.axhline(y=end - 0.5, xmin=start_rel, xmax=end_rel, **style)
        ax.axvline(x=start - 0.5, ymin=1 - start_rel, ymax=1 - end_rel, **style)
        ax.axvline(x=end - 0.5, ymin=1 - start_rel, ymax=1 - end_rel, **style)


def main():
    """Visualize the different contributions to the Gram matrix on a toy problem."""
    # setup
    manual_seed(0)
    batch_size = 10
    X = rand(batch_size, 4)
    layers = [
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

    # use a shared color limit
    vmin = min(*[c.min() for c in contributions.values()], gram.min())
    vmax = max(*[c.max() for c in contributions.values()], gram.max())

    # visualize the Gram matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")  # turn off ticks and tick labels
    im = ax.imshow(gram, vmin=vmin, vmax=vmax)
    highlight_block_diagonal(ax, dims)
    # positioning of the color bar from https://stackoverflow.com/a/43425119
    fig.colorbar(im, orientation="horizontal", location="top", pad=0.1)
    fig.savefig(path.join(FIGDIR, "gram_full.png"), bbox_inches="tight")
    plt.close(fig)

    # visualize the block-diagonal Gram matrix
    block_diag_gram = separate_into_tiles(gram, dims)
    # set off-diagonal blocks to zero
    for i, j in product(range(len(dims)), range(len(dims))):
        if i != j:
            block_diag_gram[i][j] = zeros_like(block_diag_gram[i][j])
    block_diag_gram = combine_tiles(block_diag_gram)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")  # turn off ticks and tick labels
    im = ax.imshow(block_diag_gram, vmin=vmin, vmax=vmax)
    highlight_block_diagonal(ax, dims)
    # positioning of the color bar from https://stackoverflow.com/a/43425119
    fig.savefig(path.join(FIGDIR, "gram_block_diag.png"), bbox_inches="tight")
    plt.close(fig)

    # visualize the contributions
    for child1, child2 in contributions.keys():
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis("off")  # turn off ticks and tick labels
        im = ax.imshow(contributions[(child1, child2)], vmin=vmin, vmax=vmax)
        highlight_block_diagonal(ax, dims)
        plt.savefig(
            path.join(FIGDIR, f"gram_{child1}_{child2}.png"), bbox_inches="tight"
        )
        plt.close(fig)

    # visualize the sum of terms from identical children
    diag_children = sum(contributions[(c, c)] for c in CHILDREN)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")  # turn off ticks and tick labels
    im = ax.imshow(diag_children, vmin=vmin, vmax=vmax)
    highlight_block_diagonal(ax, dims)
    plt.savefig(path.join(FIGDIR, "gram_diag_children.png"), bbox_inches="tight")
    plt.close(fig)

    # visualize the sum of terms from different children
    offdiag_children = sum(
        contributions[(c1, c2)] for c1, c2 in product(CHILDREN, CHILDREN) if c1 != c2
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")  # turn off ticks and tick labels
    im = ax.imshow(offdiag_children, vmin=vmin, vmax=vmax)
    highlight_block_diagonal(ax, dims)
    plt.savefig(path.join(FIGDIR, "gram_offdiag_children.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
