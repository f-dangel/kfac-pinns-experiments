"""Tests for kfac_pinns_exp/__init__.py."""

import time

import pytest

import kfac_pinns_exp

NAMES = ["world", "github"]
IDS = NAMES


@pytest.mark.parametrize("name", NAMES, ids=IDS)
def test_hello(name: str):
    """Test hello function.

    Args:
        name: Name to greet.
    """
    kfac_pinns_exp.hello(name)


@pytest.mark.expensive
@pytest.mark.parametrize("name", NAMES, ids=IDS)
def test_hello_expensive(name: str):
    """Expensive test of hello. Will only be run on master/main and development.

    Args:
        name: Name to greet.
    """
    time.sleep(1)
    kfac_pinns_exp.hello(name)
