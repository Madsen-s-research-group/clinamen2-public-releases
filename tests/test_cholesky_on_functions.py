from typing import Tuple

import numpy as np
import numpy.typing as npt
from scipy.optimize import rosen

from clinamen2.cmaes.params_and_state import (
    AlgorithmParameters,
    AlgorithmState,
    create_init_algorithm_state,
    create_sample_and_evaluate,
    create_sample_from_state,
    create_update_algorithm_state,
    init_default_algorithm_parameters,
)

pytest_plugins = ["pytest-datadir"]


def transposed_rosen(vec: npt.ArrayLike) -> float:
    """The function expects the transposed shape than is used here."""
    return rosen(vec.T)


def cholesky_setup(
    dimension, step_size
) -> Tuple[AlgorithmParameters, AlgorithmState, npt.ArrayLike]:
    """Helper function to initialize the algorithm"""

    # random founder
    rng = np.random.default_rng(0)
    founder = rng.random(dimension)

    # AlgorithmParameters
    parameters = init_default_algorithm_parameters(
        dimension, initial_step_size=step_size
    )

    # AlgorithmState
    init_state = create_init_algorithm_state(parameters)
    state = init_state(mean=founder)

    return parameters, state, founder


def test_exact_solution():
    """Run a low dimensional optimization to its exact solution"""
    (parameters, state, _) = cholesky_setup(dimension=2, step_size=10.0)
    update_state = create_update_algorithm_state(parameters)
    sample_individuals = create_sample_from_state(parameters)
    evaluate_loss = transposed_rosen

    sample_with_evaluate = create_sample_and_evaluate(
        sample_individuals, evaluate_loss
    )

    for g in range(5000):
        generation, state, loss = sample_with_evaluate(state)
        idx = np.argsort(loss)
        state = update_state(state, generation[idx])
        if np.isclose(loss.min(), 0.0, atol=1e-14):
            print(f"Exact solution after {g} generations.")
            break

    print(f"loss {loss.min()} for individual {generation[loss.argmin()]}")

    assert g == 106
