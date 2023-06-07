from dataclasses import asdict
from typing import Dict

import numpy as np
import pytest

from clinamen2.cmaes.params_and_state import (
    AlgorithmParameters,
    init_default_algorithm_parameters,
)


def copy_and_update_dict(dictionary: Dict, changes: Dict) -> Dict:
    outdict = dictionary.copy()
    outdict.update(changes)

    return outdict


@pytest.fixture(name="minimal_parameters_asdict")
def fixture_minimal_parameters_as_dict(minimal_parameters) -> Dict:
    """Fixture returning dict version of minimal AlgorithmParameters"""

    paramsdict = asdict(minimal_parameters)

    return paramsdict


def test_direct_initialize_no_params() -> None:
    """Test direct initialization without arguments failing"""

    with pytest.raises(TypeError):
        AlgorithmParameters(None)


def test_default_initialize(minimal_parameters):
    """Test minimal initialization"""
    assert isinstance(minimal_parameters, AlgorithmParameters)


def test_default_initialize_no_params() -> None:
    """Test default initialization without any arguments failing"""

    with pytest.raises(TypeError):
        init_default_algorithm_parameters()


@pytest.mark.parametrize(
    "dim, target_pop_size", [(10, 10), (20, 12), (100, 17)]
)
def test_default_pop_size(dim, target_pop_size):
    """Test correct implementation of default population size formula"""
    algorithm_parameters = init_default_algorithm_parameters(dimension=dim)
    assert algorithm_parameters.pop_size == target_pop_size


@pytest.mark.parametrize("value", [0, -10])
def test_nonpositive_dimension(minimal_parameters_asdict, value):
    """Test for error when dimension not positive"""

    paramsdict = copy_and_update_dict(
        minimal_parameters_asdict, {"dimension": value}
    )
    with pytest.raises(ValueError):
        AlgorithmParameters(**paramsdict)


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_nonpositive_step_size(minimal_parameters_asdict, value):
    """Test for error when initial step size not positive"""

    paramsdict = copy_and_update_dict(
        minimal_parameters_asdict, {"initial_step_size": value}
    )
    with pytest.raises(ValueError):
        AlgorithmParameters(**paramsdict)


@pytest.mark.parametrize("value", [-10, 0, 1])
def test_pop_size_less_than_two(minimal_parameters_asdict, value):
    """Test for error when population size < 2"""

    paramsdict = copy_and_update_dict(
        minimal_parameters_asdict, {"pop_size": value}
    )
    with pytest.raises(ValueError):
        AlgorithmParameters(**paramsdict)


def test_mu_out_of_bounds(minimal_parameters_asdict):
    """Test for error when initializing mu to outside [1, pop_size]"""

    lower = 0
    upper = minimal_parameters_asdict["pop_size"] + 1

    paramsdict = copy_and_update_dict(minimal_parameters_asdict, {"mu": lower})
    with pytest.raises(ValueError):
        AlgorithmParameters(**paramsdict)

    paramsdict = copy_and_update_dict(minimal_parameters_asdict, {"mu": upper})
    with pytest.raises(ValueError):
        AlgorithmParameters(**paramsdict)


def test_weights_wrong_shape(minimal_parameters_asdict):
    """Test for error when initializing with weights of wrong shape"""

    weights = np.zeros((minimal_parameters_asdict["dimension"], 2))
    paramsdict = copy_and_update_dict(
        minimal_parameters_asdict, {"weights": weights}
    )
    with pytest.raises(ValueError, match=r".*1-d.*"):
        AlgorithmParameters(**paramsdict)


def test_weights_wrong_order(minimal_parameters_asdict):
    """Test for error if first mu weights not in non-ascending order"""

    mu = minimal_parameters_asdict["mu"]
    w = minimal_parameters_asdict["weights"]
    rng = np.random.default_rng(0)
    w_pos = rng.permutation(w[:mu])
    w_neg = w[mu:]
    paramsdict = copy_and_update_dict(
        minimal_parameters_asdict, {"weights": np.concatenate([w_pos, w_neg])}
    )
    with pytest.raises(ValueError, match=r".*non-ascending.*"):
        AlgorithmParameters(**paramsdict)


def test_weights_negative(minimal_parameters_asdict):
    """Test for error if first mu weights negative"""

    mu = minimal_parameters_asdict["mu"]
    w = minimal_parameters_asdict["weights"]
    w[:mu] *= -1
    paramsdict = copy_and_update_dict(
        minimal_parameters_asdict, {"weights": w}
    )
    with pytest.raises(ValueError, match=r".*nonnegative.*"):
        AlgorithmParameters(**paramsdict)


def test_weights_wrong_sum(minimal_parameters_asdict):
    """Test for error if first mu weights do not sum to one"""

    weights, mu = (
        minimal_parameters_asdict["weights"],
        minimal_parameters_asdict["mu"],
    )
    weights[:mu] *= 1.1
    paramsdict = copy_and_update_dict(
        minimal_parameters_asdict, {"weights": weights}
    )
    with pytest.raises(ValueError, match=r".*sum to one*"):
        AlgorithmParameters(**paramsdict)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_c_sigma_out_of_bounds(minimal_parameters_asdict, value):
    """Test for error when initializing c_sigma to value outside (0, 1)"""

    paramsdict = copy_and_update_dict(
        minimal_parameters_asdict, {"c_sigma": value}
    )
    with pytest.raises(ValueError):
        AlgorithmParameters(**paramsdict)


@pytest.mark.parametrize("value", [-1.0, 0.0])
def test_d_sigma_nonpositive(minimal_parameters_asdict, value):
    """Test for error when initializing d_sigma to not > 0"""

    paramsdict = copy_and_update_dict(
        minimal_parameters_asdict, {"d_sigma": value}
    )
    with pytest.raises(ValueError):
        AlgorithmParameters(**paramsdict)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_c_m_out_of_bounds(minimal_parameters_asdict, value):
    """Test for error when initializing c_m to value outside [0, 1]"""

    paramsdict = copy_and_update_dict(
        minimal_parameters_asdict, {"c_m": value}
    )
    with pytest.raises(ValueError):
        AlgorithmParameters(**paramsdict)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_c_c_out_of_bounds(minimal_parameters_asdict, value):
    """Test for error when initializing c_c to value outside [0, 1]"""

    paramsdict = copy_and_update_dict(
        minimal_parameters_asdict, {"c_c": value}
    )
    with pytest.raises(ValueError):
        AlgorithmParameters(**paramsdict)


def test_c_1_out_of_bounds(minimal_parameters_asdict):
    """Test for error when initializing c_1 to outside [0, 1 - c_mu]"""

    lower = -0.1
    upper = 1 - minimal_parameters_asdict["c_mu"] + 0.1

    paramsdict = copy_and_update_dict(
        minimal_parameters_asdict, {"c_1": lower}
    )
    with pytest.raises(ValueError):
        AlgorithmParameters(**paramsdict)

    paramsdict = copy_and_update_dict(
        minimal_parameters_asdict, {"c_1": upper}
    )
    with pytest.raises(ValueError):
        AlgorithmParameters(**paramsdict)


def test_c_mu_out_of_bounds(minimal_parameters_asdict):
    """Test for error when initializing c_mu to outside [0, 1 - c_1]"""

    lower = -0.1
    upper = 1 - minimal_parameters_asdict["c_1"] + 0.1

    paramsdict = copy_and_update_dict(
        minimal_parameters_asdict, {"c_mu": lower}
    )
    with pytest.raises(ValueError):
        AlgorithmParameters(**paramsdict)

    paramsdict = copy_and_update_dict(
        minimal_parameters_asdict, {"c_mu": upper}
    )
    with pytest.raises(ValueError):
        AlgorithmParameters(**paramsdict)
