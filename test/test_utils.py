"""Test `kfac_pinns_exp.utils`."""

from pytest import raises
from torch import allclose, manual_seed, ones_like, rand, zeros_like

from kfac_pinns_exp.utils import bias_augmentation, exponential_moving_average


def test_exponential_moving_average():
    """Test exponential moving average function."""
    manual_seed(0)

    shape = (3, 4, 5)
    destination = rand(shape)
    update = rand(shape)

    invalid_factor = 1.1
    with raises(ValueError):
        exponential_moving_average(destination, update, invalid_factor)

    factor = 0.4
    destination_copy = destination.clone()
    exponential_moving_average(destination_copy, update, factor)
    assert allclose(factor * destination + (1 - factor) * update, destination_copy)


def test_bias_augmentation():
    """Test the bias augmentation helper."""
    manual_seed(0)

    # default value for `dim`
    y = rand(2, 3)
    y_augmented = bias_augmentation(y, 1)
    assert y_augmented.shape == (2, 4)
    assert allclose(y_augmented[:, :3], y)
    assert allclose(y_augmented[:, 3], ones_like(y_augmented[:, 3]))

    # specify non-default (negative) value for `dim`
    x = rand(3, 4, 6)
    x_augmented = bias_augmentation(x, 0, dim=-2)
    assert x_augmented.shape == (3, 5, 6)
    assert allclose(x_augmented[:, :4, :], x)
    assert allclose(x_augmented[:, 4, :], zeros_like(x_augmented[:, 4, :]))
