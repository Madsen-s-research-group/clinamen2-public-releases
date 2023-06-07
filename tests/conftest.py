"""Shared fixtures"""
from typing import Callable

import pytest

from clinamen2.cmaes.params_and_state import (
    AlgorithmParameters,
    AlgorithmState,
    create_init_algorithm_state,
    init_default_algorithm_parameters,
)

pytest_plugins = ["pytest-datadir"]


@pytest.fixture(name="minimal_parameters", scope="session")
def fixture_minimal_parameters() -> AlgorithmParameters:
    """Fixture returning minimal AlgorithmParameters"""

    return init_default_algorithm_parameters(dimension=8)


@pytest.fixture(name="init_state", scope="session")
def fixture_init_state(minimal_parameters) -> Callable:
    """Fixture creating a function for AlgorithmState initialization."""

    init_state = create_init_algorithm_state(minimal_parameters)

    return init_state


@pytest.fixture(name="minimal_state", scope="function")
def fixture_minimal_state(init_state) -> AlgorithmState:
    """Fixture returning minimal AlgorithmState"""

    return init_state()
