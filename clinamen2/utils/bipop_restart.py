"""Dataclass and functions for BIPOP-CMA-ES.

    References:

    [1] N. Hansen. Benchmarking a BI-population CMA-ES on the BBOB-2009
    function testbed. In Workshop Proceedings of the GECCO Genetic and
    Evolutionary Computation Conference. ACM, 2009.

    [2] I. Loshchilov. CMA-ES with Restarts for Solving CEC 2013 Benchmark
    Problems. IEEE Congress on Evolutionary Computation, Jun 2013, Cancun,
    Mexico. hal-00823880.
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class BIPOP:
    """Dataclass representing the current state of a BIPOP run.

    Args:
        default_pop_size: Default population size calculated by the CMA-ES.
        default_step_size: Default global variance of the Gaussian.
        random_state: Dictionary representing the state of the random
            generator (numpy.random.default_rng()).
        large_restart_counter: Number of restarts with large population.
        latest_large_pop_size: Latest used large population size.
        eval_counter: Function evaluations for large and small restarts.
            Saved in a Tuple[large, small].
    """

    default_pop_size: int
    default_step_size: float
    random_state: dict
    large_restart_counter: int
    latest_large_pop_size: int
    eval_counter: Tuple[int, int]


def bipop_init(default_pop_size, default_step_size, random_state) -> BIPOP:
    """Initialize a BIPOP object.

    The arguments are 'default' with respect to the BIPOP algorithm, not
    necessarily the CMA-ES default values.

    Args:
        default_pop_size: Default population size to start with.
        default_step_size: Default step size to start with.
        random_state: State of a numpy random number generator.

    Returns:
        Initial state of the BIPOP.
    """

    return BIPOP(
        default_pop_size=default_pop_size,
        default_step_size=default_step_size,
        random_state=random_state,
        large_restart_counter=0,
        latest_large_pop_size=0,
        eval_counter=(0, 0),
    )


def bipop_next_restart(bipop: BIPOP) -> Tuple[int, float, BIPOP]:
    """Calculate pop_size, step_size for next restart.

    Also returns an updated BIPOP. The first restart is always a large
    restart.

    Args:
        BIPOP: Input state of the BIPOP.

    Returns:
        tuple containing

            - Population size for next restart.
            - Step size for next restart.
            - New state of the BIPOP.
    """
    large_restart_counter = bipop.large_restart_counter
    latest_large_pop_size = bipop.latest_large_pop_size
    random_state = bipop.random_state
    if bipop.eval_counter[0] <= bipop.eval_counter[1]:
        large_restart_counter += 1
        pop_size = 2**large_restart_counter * bipop.default_pop_size
        latest_large_pop_size = pop_size
        step_size = bipop.default_step_size
    else:
        rng = np.random.default_rng()
        rng.bit_generator.state = bipop.random_state
        pop_size = int(
            np.floor(
                bipop.default_pop_size
                * (0.5 * latest_large_pop_size / bipop.default_pop_size)
                ** (rng.random() ** 2)
            )
        )
        step_size = bipop.default_step_size * 10 ** (-2.0 * rng.random())
        random_state = rng.bit_generator.__getstate__()
    new_bipop = BIPOP(
        default_pop_size=bipop.default_pop_size,
        default_step_size=bipop.default_step_size,
        random_state=random_state,
        large_restart_counter=large_restart_counter,
        latest_large_pop_size=latest_large_pop_size,
        eval_counter=bipop.eval_counter,
    )

    return pop_size, step_size, new_bipop


def bipop_update(bipop: BIPOP, new_evals: int) -> BIPOP:
    """Returns a new BIPOP with updated eval_counter.

    The lower counter will be updated.

    Args:
        bipop: Input state of the BIPOP.
        new_evals: Evaluation count performed since input state.

    Returns:
        New state of the BIPOP.
    """

    if bipop.eval_counter[0] <= bipop.eval_counter[1]:
        eval_counter = (
            bipop.eval_counter[0] + new_evals,
            bipop.eval_counter[1],
        )
    else:
        eval_counter = (
            bipop.eval_counter[0],
            bipop.eval_counter[1] + new_evals,
        )

    return BIPOP(
        default_pop_size=bipop.default_pop_size,
        default_step_size=bipop.default_step_size,
        random_state=bipop.random_state,
        large_restart_counter=bipop.large_restart_counter,
        latest_large_pop_size=bipop.latest_large_pop_size,
        eval_counter=eval_counter,
    )
