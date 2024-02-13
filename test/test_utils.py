"""Test `kfac_pinns_exp.utils`."""

from pytest import raises
from torch import allclose, manual_seed, rand

from kfac_pinns_exp.utils import exponential_moving_average


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
