"""Test cases for the standard termination criteria."""
from collections import deque

import numpy as np
import pytest

from clinamen2.cmaes.cmaes_criteria import (
    ConditionCovCriterion,
    ConditionCovState,
    EqualFunValuesCriterion,
    EqualFunValuesState,
    TolXUpCriterion,
    TolXUpState,
)

pytest_plugins = ["pytest-datadir"]


@pytest.fixture(name="condition_cov_criterion")
def fixture_condition_cov_criterion(
    minimal_parameters,
) -> ConditionCovCriterion:
    """Fixture returning default ConditionCovCriterion"""
    return ConditionCovCriterion(parameters=minimal_parameters)


@pytest.fixture(name="equal_fun_values_criterion")
def fixture_equal_fun_values_criterion(
    minimal_parameters,
) -> EqualFunValuesCriterion:
    """Fixture returning default EqualFunValuesCriterion"""
    return EqualFunValuesCriterion(parameters=minimal_parameters)


@pytest.fixture(name="tolxup_criterion")
def fixture_tolxup_criterion(
    minimal_parameters,
) -> TolXUpCriterion:
    """Fixture returning default TolXUpCriterion"""
    return TolXUpCriterion(parameters=minimal_parameters)


@pytest.mark.parametrize("threshold", [None, 1e10])
def test_create_condition_cov_criterion(
    minimal_parameters, condition_cov_criterion, threshold
) -> None:
    """Test the constructor."""
    if threshold is None:
        crit = condition_cov_criterion
        assert crit.threshold == 1e14
    else:
        crit = ConditionCovCriterion(
            parameters=minimal_parameters, threshold=threshold
        )
        assert crit.threshold == threshold


def test_init_condition_cov_criterion(condition_cov_criterion) -> None:
    """Test the init function."""
    crit_state = condition_cov_criterion.init()
    assert isinstance(crit_state, ConditionCovState)


def test_update_condition_cov_criterion(
    minimal_state, condition_cov_criterion
) -> None:
    """Test the update function."""
    crit_state = condition_cov_criterion.update(
        criterion_state=None, state=minimal_state, population=None, loss=None
    )
    assert crit_state.cond == 1.0


@pytest.mark.parametrize("cond", [1.0, 1e15])
def test_met_condition_cov_criterion(condition_cov_criterion, cond) -> None:
    """Test met function."""
    crit_state = ConditionCovState(cond=cond)
    if cond == 1.0:
        assert not condition_cov_criterion.met(criterion_state=crit_state)
    elif cond == 1e15:
        assert condition_cov_criterion.met(criterion_state=crit_state)


@pytest.mark.parametrize("atol", [None, 1e-10])
def test_create_equal_fun_values_criterion(
    minimal_parameters, equal_fun_values_criterion, atol
) -> None:
    """Test the constructor."""
    if atol is None:
        crit = equal_fun_values_criterion
        assert crit.atol == 1e-15
        assert crit.generation_span == 34
    else:
        crit = EqualFunValuesCriterion(
            parameters=minimal_parameters, atol=atol
        )
        assert crit.atol == atol
    # default generation_span was already tested together with atol None
    crit = EqualFunValuesCriterion(
        parameters=minimal_parameters, generation_span=10
    )
    assert crit.generation_span == 10


def test_init_equal_fun_values_criterion(equal_fun_values_criterion) -> None:
    """Test the init function."""
    crit_state = equal_fun_values_criterion.init()
    assert isinstance(crit_state, EqualFunValuesState)


@pytest.mark.parametrize("elements", [5, 34])
def test_update_equal_fun_values_criterion(
    minimal_state, equal_fun_values_criterion, elements
) -> None:
    """Test the update function."""
    losses = np.asarray(
        [10.0] * 10
    )  # any value would work. need to be 10 members though.
    fun_values = deque([1.0] * elements, maxlen=34)
    fun_values[0] = 0.5
    crit_state = EqualFunValuesState(fun_values=fun_values)
    crit_state = equal_fun_values_criterion.update(
        criterion_state=crit_state,
        state=minimal_state,
        population=None,
        loss=losses,
    )
    if elements == 5:
        assert len(crit_state.fun_values) == elements + 1
    elif elements == 34:
        assert len(crit_state.fun_values) == elements
        assert crit_state.fun_values[0] == 1.0
    assert crit_state.fun_values[-1] == 10.0


def test_met_equal_fun_values_criterion(equal_fun_values_criterion) -> None:
    """Test met function."""
    fun_values = [1.0] * 34
    crit_state = EqualFunValuesState(fun_values)
    assert equal_fun_values_criterion.met(criterion_state=crit_state)

    fun_values[0] = 1.001
    crit_state = EqualFunValuesState(fun_values)
    assert not equal_fun_values_criterion.met(criterion_state=crit_state)


@pytest.mark.parametrize("threshold", [None, 1e3])
def test_create_tolxup_criterion(
    minimal_parameters, tolxup_criterion, threshold
) -> None:
    """Test the constructor."""
    if threshold is None:
        crit = tolxup_criterion
        assert crit.threshold == 1e4
        assert not crit.interpolative
    else:
        crit = TolXUpCriterion(
            parameters=minimal_parameters,
            threshold=threshold,
            interpolative=True,
        )
        assert crit.threshold == threshold
        assert crit.interpolative


def test_init_tolxup_criterion(tolxup_criterion) -> None:
    """Test the init function."""
    crit_state = tolxup_criterion.init()
    assert isinstance(crit_state, TolXUpState)


@pytest.mark.parametrize("compare_to", [None, 2.0])
@pytest.mark.parametrize("interpolative", [True, False])
def test_update_tolxup_criterion(
    minimal_parameters,
    minimal_state,
    tolxup_criterion,
    compare_to,
    interpolative,
) -> None:
    """Test the update function."""
    if interpolative:
        crit = TolXUpCriterion(
            parameters=minimal_parameters, interpolative=True
        )
    else:
        crit = tolxup_criterion
    crit_state = TolXUpState(compare_to=compare_to, latest_diff=1.0)
    crit_state = crit.update(
        criterion_state=crit_state,
        state=minimal_state,
        population=None,
        loss=None,
    )
    assert crit_state.compare_to == 1.0
    if compare_to is None:
        assert crit_state.latest_diff == 0.0
    elif compare_to == 1.0:
        assert crit_state.latest_diff == 1.0


def test_met_tolxup_criterion(tolxup_criterion) -> None:
    """Test met function."""
    crit_state = TolXUpState(latest_diff=1.0)
    assert not tolxup_criterion.met(criterion_state=crit_state)

    crit_state = TolXUpState(latest_diff=1e5)
    assert tolxup_criterion.met(criterion_state=crit_state)
