"""Classes and functions implementing the CMA-ES termination criteria."""
from abc import ABC, abstractmethod
from typing import NamedTuple, Sequence

import numpy.typing as npt

from clinamen2.cmaes.params_and_state import (
    AlgorithmParameters,
    AlgorithmState,
)


class StaleLossState(NamedTuple):
    """NamedTuple to keep track of the state of a criterion.

    Args:
        counter: A variable to keep track of relevant steps.
        compare_to: A reference value to compare to.
    """

    counter: int = 0
    compare_to: float = None


class CriteriaCombinationState(NamedTuple):
    """NamedTuple to keep track of a tuple of criterion states.

    Args:
        criteria_states: Tuple containing criteria states.
    """

    criteria_states: tuple


# derived state NamedTuples
StaleStepState = StaleLossState
StaleStdState = StaleLossState
CriteriaAndState = CriteriaCombinationState
CriteriaOrState = CriteriaCombinationState


class Criterion(ABC):
    """Abstract base class for termination criteria.

    Args:
        parameters: Initial, immutable parameters of the CMA-ES run.
    """

    def __init__(self, parameters: AlgorithmParameters):
        """Constructor"""
        self.parameters = parameters

    @abstractmethod
    def init(self) -> NamedTuple:
        """
        Initialize the associated state.

        Returns:
            The initial state of the Criterion.
        """

    @abstractmethod
    def update(
        self,
        criterion_state: NamedTuple,
        state: AlgorithmState,
        population: npt.ArrayLike,
        loss: npt.ArrayLike,
    ) -> NamedTuple:
        """Function to update and return the Criterion

        Args:
            criterion_state: Current state of the Criterion.
            state: Current state of the evolution.
            population: Current generation of individuals.
            loss: Loss of each of the current individuals.

        Returns:
            The updated state of the Criterion.
        """

    @abstractmethod
    def met(self, criterion_state: NamedTuple) -> bool:
        """

        Args:
            criterion_state: State of criterion to base decision on.

        Returns:
            True if the Criterion is fulfilled, False if not.
        """


class StaleLossCriterion(Criterion):
    """Class that implements a termination criterion of the CMA-ES.

    Takes the loss trajectory into account. If the loss is stale for a
    given number of generations, the criterion is met.

    Args:
        parameters: Initial, immutable parameters of the CMA-ES run.
        threshold: Difference up to which two different loss values are
            considered equal.
        generations: Number of generations for which the loss has to be
            stale for the criterion to be met.
    """

    def __init__(
        self,
        parameters: AlgorithmParameters,
        threshold: float,
        generations: int,
    ):
        self.threshold = threshold
        self.generations = generations
        super().__init__(parameters=parameters)

    def init(self) -> StaleLossState:
        """Initialize the associated CriterionState.

        Use base CriterionState and set counter to zero.
        """
        return StaleLossState(counter=0, compare_to=None)

    def update(
        self,
        criterion_state: StaleLossState,
        state: AlgorithmState,
        population: npt.ArrayLike,
        loss: npt.ArrayLike,
    ) -> StaleLossState:
        """Function to update and return the Criterion

        Args:
            criterion_state: Current state of the Criterion.
            state: Current state of the evolution.
            population: Current generation of individuals.
            loss: Loss of each of the current individuals.

        Returns:
            The updated state of the Criterion.
        """
        compare_to = loss.min()
        if (
            criterion_state.compare_to is not None
            and abs(criterion_state.compare_to - loss.min()) < self.threshold
        ):
            counter = criterion_state.counter + 1
        else:
            counter = 0
        return StaleLossState(counter=counter, compare_to=compare_to)

    def met(self, criterion_state: StaleLossState) -> bool:
        """

        Args:
            criterion_state: State of criterion to base decision on.

        Returns:
            True if the Criterion is fulfilled, False if not.
        """
        return criterion_state.counter >= self.generations


class StaleStepCriterion(Criterion):
    """Class that implements a termination criterion of the CMA-ES.

    Takes the step size trajectory into account. If the step size is stale for
    a given number of generations, the criterion is met.

    Args:
        parameters: Initial, immutable parameters of the CMA-ES run.
        threshold: Difference up to which two different step sizes are
            considered equal.
        generations: Number of generations for which the step size needs to
           be stale for the criterion to be met.
    """

    def __init__(
        self,
        parameters: AlgorithmParameters,
        threshold: float,
        generations: int,
    ):
        self.threshold = threshold
        self.generations = generations
        super().__init__(parameters=parameters)

    def init(self) -> StaleStepState:
        """Initialize the associated CriterionState.

        Use base CriterionState and set counter to zero.
        """
        return StaleStepState(counter=0, compare_to=None)

    def update(
        self,
        criterion_state: StaleStepState,
        state: AlgorithmState,
        population: npt.ArrayLike,
        loss: npt.ArrayLike,
    ) -> StaleStepState:
        """Function to update and return the Criterion

        Args:
            criterion_state: Current state of the Criterion.
            state: Current state of the evolution.
            population: Current generation of individuals.
            loss: Loss of each of the current individuals.

        Returns:
            The updated state of the Criterion.
        """
        compare_to = state.step_size
        if (
            criterion_state.compare_to is not None
            and abs(criterion_state.compare_to - state.step_size)
            < self.threshold
        ):
            counter = criterion_state.counter + 1
        else:
            counter = 0
        return StaleStepState(counter=counter, compare_to=compare_to)

    def met(self, criterion_state: StaleStepState) -> bool:
        """

        Args:
            criterion_state: State of criterion to base decision on.

        Returns:
            True if the Criterion is fulfilled, False if not.
        """
        return criterion_state.counter >= self.generations


class StaleStdCriterion(Criterion):
    """Class that implements a termination criterion of the CMA-ES.

    Takes the standard deviation within generations into account. If the std
    is below a threshold for a given number of generations, the criterion is
    met.

    Args:
        parameters: Initial, immutable parameters of the CMA-ES run.
        threshold: Threshold value for std to fall below.
        generations: Number of generations for which std needs to remain below
            threshold for the criterion to be met.
    """

    def __init__(
        self,
        parameters: AlgorithmParameters,
        threshold: float,
        generations: int,
    ):
        self.threshold = threshold
        self.generations = generations
        super().__init__(parameters=parameters)

    def init(self) -> StaleStdState:
        """Initialize the associated CriterionState.

        Use base CriterionState and set counter to zero.
        """
        return StaleStdState(counter=0, compare_to=None)

    def update(
        self,
        criterion_state: StaleStdState,
        state: AlgorithmState,
        population: npt.ArrayLike,
        loss: npt.ArrayLike,
    ) -> StaleLossState:
        """Function to update and return the Criterion

        Args:
            criterion_state: Current state of the Criterion.
            state: Current state of the evolution.
            population: Current generation of individuals.
            loss: Loss of each of the current individuals.

        Returns:
            The updated state of the Criterion.
        """
        if loss.std() < self.threshold:
            counter = criterion_state.counter + 1
        else:
            counter = 0

        # this criterion does not use `compare_to`
        return StaleLossState(counter=counter, compare_to=0.0)

    def met(self, criterion_state: StaleStdState) -> bool:
        """

        Args:
            criterion_state: State of criterion to base decision on.

        Returns:
            True if the Criterion is fulfilled, False if not.
        """
        return criterion_state.counter >= self.generations


class CriteriaCombination(Criterion, ABC):
    """Abstract class that combines criteria.

    Args:
        parameters: Initial, immutable parameters of the CMA-ES run.
        criteria: Sequence of criteria to be combined.
    """

    def __init__(
        self,
        parameters: AlgorithmParameters,
        criteria: Sequence,
    ):
        self.criteria = criteria
        super().__init__(parameters=parameters)

    def init(self) -> Sequence:
        """Initialize the associated CriterionState instances."""
        criteria_states = []
        for criterion in self.criteria:
            criteria_states.append(criterion.init())
        return CriteriaCombinationState(criteria_states=tuple(criteria_states))

    def update(
        self,
        criterion_state: CriteriaCombinationState,
        state: AlgorithmState,
        population: npt.ArrayLike,
        loss: npt.ArrayLike,
    ) -> CriteriaCombinationState:
        """Function to update and return the criteria

        Args:
            criterion_state: NamedTuple containing tuple of criteria states.
            state: Current state of the evolution.
            population: Current generation of individuals.
            loss: Loss of each of the current individuals.

        Returns:
            NamedTuple with tuple of the updated states of the criteria.
        """
        criteria_states = [
            criterion.update(
                criterion_state=criterion_state.criteria_states[c],
                state=state,
                population=population,
                loss=loss,
            )
            for c, criterion in enumerate(self.criteria)
        ]
        return CriteriaCombinationState(criteria_states=tuple(criteria_states))

    @abstractmethod
    def met(self, criterion_state: CriteriaCombinationState) -> bool:
        """

        Args:
            criterion_state: NamedTuple containing tuple of criteria states.

        Returns:
            True if criteria combination is fulfilled, False otherwise.
        """


class CriteriaAnd(CriteriaCombination):
    """Class that combines criteria that all have to be fulfilled.

    Evaluates multiple criteria (instances of Criterion). All have to meet
    their respective parameters for the CriterionAnd to be met.

    Args:
        parameters: Initial, immutable parameters of the CMA-ES run.
        criteria: Sequence of criteria to be combined.
    """

    def met(self, criterion_state: CriteriaAndState) -> bool:
        """

        Args:
            criterion_state: NamedTuple containing tuple of criteria states.

        Returns:
            True if all criteria are fulfilled, False if any is not.
        """
        for c, criterion in enumerate(self.criteria):
            if not criterion.met(criterion_state.criteria_states[c]):
                return False
        return True


class CriteriaOr(CriteriaCombination):
    """Class that combines criteria were one has to be fulfilled.

    Evaluates multiple criteria (instances of Criterion). Any one has to meet
    their respective parameters for the CriterionOr to be met.

    Args:
        parameters: Initial, immutable parameters of the CMA-ES run.
        criteria: Sequence of criteria to be combined.
    """

    def met(self, criterion_state: CriteriaOrState) -> bool:
        """

        Args:
            criterion_state: NamedTuple containing tuple of criteria states.

        Returns:
            True if any criteria is fulfilled, False if none are.
        """
        for c, criterion in enumerate(self.criteria):
            if criterion.met(criterion_state.criteria_states[c]):
                return True
        return False
