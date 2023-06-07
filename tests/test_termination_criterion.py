"""Testcases for clinamen2.cmaes.termination_criterion"""
import numpy as np
import pytest

from clinamen2.cmaes.termination_criterion import (
    CriteriaAnd,
    CriteriaAndState,
    CriteriaOr,
    CriteriaOrState,
    StaleLossCriterion,
    StaleLossState,
    StaleStepCriterion,
    StaleStepState,
)

pytest_plugins = ["pytest-datadir"]


@pytest.fixture(name="stale_loss_criterion")
def fixture_stale_loss_criterion(minimal_parameters) -> StaleLossCriterion:
    """Fixture returning default StaleLossCriterion"""
    return StaleLossCriterion(
        parameters=minimal_parameters, threshold=1e-5, generations=10
    )


@pytest.fixture(name="stale_step_criterion")
def fixture_stale_step_criterion(minimal_parameters) -> StaleStepCriterion:
    """Fixture returning default StaleStepCriterion"""
    return StaleStepCriterion(
        parameters=minimal_parameters, threshold=1e-5, generations=10
    )


@pytest.fixture(name="termination_criteria")
def fixture_termination_criteria(minimal_parameters) -> list:
    """Fixture returning example termination criteria list"""
    termination_criteria = [
        StaleStepCriterion(
            parameters=minimal_parameters, threshold=1e-10, generations=10
        ),
        StaleLossCriterion(
            parameters=minimal_parameters, threshold=1e-5, generations=10
        ),
    ]
    return termination_criteria


@pytest.fixture(name="dummy_population")
def fixture_dummy_population(minimal_parameters):
    """Fixture returning some dummy population of the correct shape."""
    return np.zeros(
        shape=(minimal_parameters.pop_size, minimal_parameters.dimension)
    )


def test_create_stale_loss_criterion(stale_loss_criterion) -> None:
    """Test constructor of StaleLossCriterion"""
    assert isinstance(stale_loss_criterion, StaleLossCriterion)


def test_init_stale_loss_state(stale_loss_criterion) -> None:
    """Test init() function of StaleLossCriterion"""
    state = stale_loss_criterion.init()
    assert isinstance(state, StaleLossState)
    assert state.counter == 0
    assert state.compare_to is None


def test_init_stale_step_state(stale_step_criterion) -> None:
    """Test init() function of StaleLossCriterion"""
    state = stale_step_criterion.init()
    assert isinstance(state, StaleStepState)
    assert state.counter == 0
    assert state.compare_to is None


@pytest.mark.parametrize("compare_to", [1e-10, 10.0])
def test_update_stale_loss_state(
    stale_loss_criterion,
    dummy_population,
    minimal_parameters,
    minimal_state,
    compare_to,
) -> None:
    """Test update() function of StaleLossCriterion"""
    state = StaleLossState(counter=2, compare_to=compare_to)
    loss = np.zeros(shape=(minimal_parameters.pop_size))
    new_state = stale_loss_criterion.update(
        criterion_state=state,
        state=minimal_state,
        population=dummy_population,
        loss=loss,
    )
    assert new_state.compare_to == 0.0
    if compare_to == 1e-10:
        assert new_state.counter == 3
    elif compare_to == 10.0:
        assert new_state.counter == 0


@pytest.mark.parametrize("counter", [0, 10])
def test_met_stale_loss_state(stale_loss_criterion, counter) -> None:
    """Test met() function of StaleLossCriterion"""
    state = StaleLossState(counter=counter, compare_to=1e-10)
    met = stale_loss_criterion.met(state)
    if counter == 0:
        assert not met
    elif counter == 10:
        assert met


def test_create_criteria_or(minimal_parameters, termination_criteria) -> None:
    """Test constructor of CriteriaOr"""
    criteria_or = CriteriaOr(
        parameters=minimal_parameters, criteria=termination_criteria
    )
    assert isinstance(criteria_or, CriteriaOr)


def test_create_criteria_and(minimal_parameters, termination_criteria) -> None:
    """Test constructor of CriteriaAnd"""
    criteria_or = CriteriaAnd(
        parameters=minimal_parameters, criteria=termination_criteria
    )
    assert isinstance(criteria_or, CriteriaAnd)


def test_init_criteria_or(minimal_parameters, termination_criteria) -> None:
    """Test init() function of CriteriaOr"""
    criteria_or = CriteriaOr(
        parameters=minimal_parameters, criteria=termination_criteria
    )
    state = criteria_or.init()
    assert isinstance(state, CriteriaOrState)
    assert len(state.criteria_states) == 2


def test_init_criteria_and(minimal_parameters, termination_criteria) -> None:
    """Test init() function of CriteriaAnd"""
    criteria_or = CriteriaAnd(
        parameters=minimal_parameters, criteria=termination_criteria
    )
    state = criteria_or.init()
    assert isinstance(state, CriteriaOrState)
    assert len(state.criteria_states) == 2


def test_update_criteria_or(
    minimal_parameters, minimal_state, dummy_population, termination_criteria
) -> None:
    """Test update() function of CriteriaOr"""
    criteria_or = CriteriaOr(
        parameters=minimal_parameters, criteria=termination_criteria
    )
    loss = np.zeros(shape=(minimal_parameters.pop_size))
    state = CriteriaOrState(
        criteria_states=(
            StaleStepState(counter=2, compare_to=1.0),
            StaleLossState(counter=2, compare_to=1.0),
        )
    )
    state = criteria_or.update(
        criterion_state=state,
        state=minimal_state,
        population=dummy_population,
        loss=loss,
    )
    assert isinstance(state, CriteriaOrState)
    assert len(state.criteria_states) == 2


def test_update_criteria_and(
    minimal_parameters, minimal_state, dummy_population, termination_criteria
) -> None:
    """Test update() function of CriteriaAnd"""
    criteria_and = CriteriaAnd(
        parameters=minimal_parameters, criteria=termination_criteria
    )
    loss = np.zeros(shape=(minimal_parameters.pop_size))
    state = CriteriaAndState(
        criteria_states=(
            StaleStepState(counter=2, compare_to=1.0),
            StaleLossState(counter=2, compare_to=1.0),
        )
    )
    state = criteria_and.update(
        criterion_state=state,
        state=minimal_state,
        population=dummy_population,
        loss=loss,
    )
    assert isinstance(state, CriteriaAndState)
    assert len(state.criteria_states) == 2


@pytest.mark.parametrize("counters", [[10, 10], [10, 0], [10, 0], [0, 0]])
def test_met_criteria_or(
    termination_criteria, minimal_parameters, counters
) -> None:
    """Test met() function of CriteriaOr"""
    criteria_or = CriteriaOr(
        parameters=minimal_parameters, criteria=termination_criteria
    )
    state = CriteriaOrState(
        criteria_states=(
            StaleStepState(counter=counters[0], compare_to=1e-10),
            StaleLossState(counter=counters[1], compare_to=1e-10),
        )
    )
    met = criteria_or.met(state)
    if counters == [10, 10] or counters == [10, 0] or counters == [0, 10]:
        assert met
    elif counters == [0, 0]:
        assert not met


@pytest.mark.parametrize("counters", [[10, 10], [10, 0], [10, 0], [0, 0]])
def test_met_criteria_and(
    termination_criteria, minimal_parameters, counters
) -> None:
    """Test met() function of CriteriaOr"""
    criteria_and = CriteriaAnd(
        parameters=minimal_parameters, criteria=termination_criteria
    )
    state = CriteriaAndState(
        criteria_states=(
            StaleStepState(counter=counters[0], compare_to=1e-10),
            StaleLossState(counter=counters[1], compare_to=1e-10),
        )
    )
    met = criteria_and.met(state)
    if counters == [10, 0] or counters == [0, 10] or counters == [0, 0]:
        assert not met
    elif counters == [10, 10]:
        assert met
