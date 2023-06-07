""" Implementation of the termination criteria as described in

    References:

    [1] N. Hansen, 2016, arXiv:1604.00772 [cs.LG]
"""
from collections import deque
from typing import NamedTuple

import numpy as np
import numpy.linalg as lin
import numpy.typing as npt
import scipy.linalg

from clinamen2.cmaes.params_and_state import (
    AlgorithmParameters,
    AlgorithmState,
)
from clinamen2.cmaes.termination_criterion import Criterion


class ConditionCovState(NamedTuple):
    """NamedTuple to keep track of condition number

    Args:
        cond: Condition number of C.
    """

    cond: float = None


class ConditionCovCriterion(Criterion):
    """ConditionCov criterion

    Stop if the condition number of the covariance matrix exceeds 1e14.

    Args:
        parameters: Initial, immutable parameters of the CMA-ES run.
        threshold: Upper limit for accepted condition number. Default is 1e14.
    """

    def __init__(
        self, parameters: AlgorithmParameters, threshold: float = 1e14
    ):
        self.threshold = threshold
        super().__init__(parameters=parameters)

    def init(self) -> ConditionCovState:
        """Initialize the associated NamedTuple."""
        return ConditionCovState(cond=None)

    def update(
        self,
        criterion_state: ConditionCovState,
        state: AlgorithmState,
        population: npt.ArrayLike,
        loss: npt.ArrayLike,
    ) -> ConditionCovState:
        """Return an updated associated NamedTuple.

        Args:
            criterion_state: The associated NamedTuple representing
                the current stopping criterion state.
            state: New algorithm state based on which to update the stopping
                criterion.
            population: New population based on which to update the stopping
                criterion.
            loss: New loss values based on which to update the stopping
                criterion.
        """
        cond = lin.cond(state.cholesky_factor) ** 2
        return ConditionCovState(cond=cond)

    def met(self, criterion_state: ConditionCovState) -> bool:
        """Decide if the stopping criterion is met.

        Args:
            criterion_state: The associated NamedTuple representing
                the current stopping criterion state.
        """
        return criterion_state.cond > self.threshold


class EqualFunValuesState(NamedTuple):
    """NamedTuple to keep track of function value staleness.

    Args:
        fun_values: Function values of past generations to be taken into
            account.
    """

    fun_values: deque = None


class EqualFunValuesCriterion(Criterion):
    """EqualFunValues criterion

    Stop if the range of the loss within a certain range of generations is
    close to zero.

    Args:
        parameters: The algorithm parameters.
        generation_span: Number of generations over which the range of function
            values is to be taken into account.
            Default is 10 + ceil(30 dimension / pop_size).
        atol: Tolerance for 'zero'. Default is 1e-15.

    """

    def __init__(
        self,
        parameters: AlgorithmParameters,
        generation_span: int = 0,
        atol: float = 1e-15,
    ):
        self.generation_span = int(
            generation_span
            if generation_span > 1
            else 10 + np.ceil(30 * parameters.dimension / parameters.pop_size)
        )
        self.atol = atol
        super().__init__(parameters=parameters)

    def init(self) -> EqualFunValuesState:
        """Initialize the associated NamedTuple."""
        print(f"span is {self.generation_span}")
        return EqualFunValuesState(
            fun_values=deque(maxlen=self.generation_span)
        )

    def update(
        self,
        criterion_state: EqualFunValuesState,
        state: AlgorithmState,
        population: npt.ArrayLike,
        loss: npt.ArrayLike,
    ) -> EqualFunValuesState:
        """Return an updated associated NamedTuple.

        Args:
            criterion_state: The associated NamedTuple representing
                the current stopping criterion state.
            state: New algorithm state based on which to update the stopping
                criterion.
            population: New population based on which to update the stopping
                criterion.
            loss: New loss values based on which to update the stopping
                criterion.
        """
        vals_deque = criterion_state.fun_values
        vals_deque.append(loss.min())
        return EqualFunValuesState(fun_values=vals_deque)

    def met(self, criterion_state: EqualFunValuesState) -> bool:
        """Decide if the stopping criterion is met.

        Args:
            criterion_state: The associated NamedTuple representing
                the current stopping criterion state.
        """
        fun_values = np.asarray(criterion_state.fun_values)
        return (
            len(criterion_state.fun_values) == self.generation_span
            and abs(fun_values.max() - fun_values.min()) < self.atol
        )


class TolXUpState(NamedTuple):
    """NamedTuple to keep track of function value staleness.

    Args:
        compare_to: Previous value of tolxup saved for comparison.
        latest_diff: Latest calculated absolute difference in tolxup values.
    """

    compare_to: float = None
    latest_diff: float = None


class TolXUpCriterion(Criterion):
    """TolXUp criterion

    Stop if sigma times max(diag(D)) exceeds a threshold when compared to the
    previous generation.

    Args:
        parameters: Initial, immutable parameters of the CMA-ES run.
        threshold: Upper limit for accepted difference. Default is 1e4.
        interpolative: Control how the matrix norm is calculated.
            True: 'scipy.linalg.interpolative.estimate_spectral_norm'
            False: 'scipy.linalg.norm'
            Default is False. Use True for large matrices.
    """

    def __init__(
        self,
        parameters: AlgorithmParameters,
        threshold: float = 1e4,
        interpolative: bool = False,
    ):
        self.threshold = threshold
        self.interpolative = interpolative
        if interpolative:
            self.norm_func = scipy.linalg.interpolative.estimate_spectral_norm
            self.norm_params = {}
        else:
            self.norm_func = scipy.linalg.norm
            self.norm_params = {"axis": (0, 1), "ord": 2}
        super().__init__(parameters=parameters)

    def init(self) -> TolXUpState:
        """Initialize the associated NamedTuple."""
        return TolXUpState()

    def update(
        self,
        criterion_state: TolXUpState,
        state: AlgorithmState,
        population: npt.ArrayLike,
        loss: npt.ArrayLike,
    ) -> TolXUpState:
        """Return an updated associated NamedTuple.

        Args:
            criterion_state: The associated NamedTuple representing
                the current stopping criterion state.
            state: New algorithm state based on which to update the stopping
                criterion.
            population: New population based on which to update the stopping
                criterion.
            loss: New loss values based on which to update the stopping
                criterion.
        """
        compare_to = state.step_size * self.norm_func(
            state.cholesky_factor, **self.norm_params
        )
        if criterion_state.compare_to is None:
            latest_diff = 0.0
        else:
            latest_diff = abs(criterion_state.latest_diff - compare_to)
        return TolXUpState(compare_to=compare_to, latest_diff=latest_diff)

    def met(self, criterion_state: TolXUpState) -> bool:
        """Decide if the stopping criterion is met.

        Args:
            criterion_state: The associated NamedTuple representing
                the current stopping criterion state.
        """
        return criterion_state.latest_diff > self.threshold
