import pickle
from typing import Callable

import dask
import numpy as np
import numpy.typing as npt
import pytest

from clinamen2.cmaes.params_and_state import (
    AlgorithmState,
    calculate_new_mean,
    create_init_algorithm_state,
    create_resample_and_evaluate,
    create_sample_and_evaluate,
    create_sample_and_run,
    create_sample_and_sequential_evaluate,
    create_sample_from_state,
    create_update_algorithm_state,
    rank_one_update,
)
from clinamen2.runner.basic_runner import FunctionRunner
from clinamen2.test_functions import create_sphere_function
from clinamen2.utils.file_handling import CMAFileHandler
from tests.aux_functions import (
    pretend_to_fail_rosen_with_info,
    single_unreliable_rosen_with_info,
    transposed_rosen,
    transposed_rosen_with_info,
    transposed_rosen_with_info_and_inputs,
    unreliable_rosen,
    unreliable_rosen_internal_rng,
)

pytest_plugins = ["pytest-datadir"]


@pytest.fixture(name="dask_setup", autouse=True)
def fixture_dask_setup(shared_datadir) -> None:
    dask.config.set(
        {
            "distributed.worker.use-file-locking": False,
            "temporary-directory": shared_datadir,
        }
    )


@pytest.fixture(name="loss_results")
def fixture_loss_results() -> npt.ArrayLike:
    return np.array(
        [
            280.99429711,
            5240.33617504,
            1104.18603571,
            774.31598446,
            271.37282199,
            418.39748095,
            1465.09102284,
            2260.96431105,
            3553.43320145,
            3708.48628528,
        ]
    )


@pytest.fixture(name="update_state", scope="session")
def fixture_update_state(minimal_parameters) -> Callable:
    """Fixture creating a function for AlgorithmState update."""

    update_state = create_update_algorithm_state(minimal_parameters)

    return update_state


@pytest.fixture(name="zeros_population", scope="function")
def fixture_zeros_population(minimal_parameters) -> npt.ArrayLike:
    """Fixture creating a population of individuals with all zeros."""

    return np.zeros(
        shape=(minimal_parameters.pop_size, minimal_parameters.dimension)
    )


@pytest.fixture(name="simple_population", scope="function")
def fixture_simple_population(
    minimal_parameters, minimal_state
) -> npt.ArrayLike:
    """Fixture sampling a population of individuals from state."""

    simple_sample = create_sample_from_state(minimal_parameters)
    simple_population, _ = simple_sample(minimal_state)

    return simple_population


def test_initialize(minimal_state) -> None:
    """Test minimal initializiation"""

    assert isinstance(minimal_state, AlgorithmState)


def test_initialize_no_params() -> None:
    """Test initializiation without parameters"""

    with pytest.raises(TypeError):
        create_init_algorithm_state(None)


def test_initialize_rng(minimal_state) -> None:
    """Test initializiation of random number generator"""

    rng = np.random.default_rng()
    rng.bit_generator.state = minimal_state.random_state
    assert rng.integers(100) == 85


def test_initialize_mean_wrong_shape(init_state) -> None:
    """Test initialization with wrong shape of mean"""

    with pytest.raises(AttributeError):
        init_state(mean=np.zeros(1))


def test_initialize_cholesky_factor_wrong_shape(init_state) -> None:
    """Test initialization with wrong shape of cholesky_factor"""

    with pytest.raises(AttributeError):
        init_state(cholesky_factor=np.zeros(shape=(10, 2)))


def test_initialize_generation_invalid(init_state) -> None:
    """Test initializiation with invalid generation"""

    with pytest.raises(TypeError):
        init_state(generation=0.5)
    with pytest.raises(ValueError):
        init_state(generation=-1)


def test_initialize_psigma_wrong_shape(init_state) -> None:
    """Test initialization with wrong shape of p_sigma"""

    with pytest.raises(AttributeError):
        init_state(p_sigma=np.zeros(1))


def test_initialize_pcov_wrong_shape(init_state) -> None:
    """Test initialization with wrong shape of p_c"""

    with pytest.raises(AttributeError):
        init_state(p_c=np.zeros(1))


def test_init_algorithm_state(minimal_parameters) -> None:
    """Test initializiation via designated function"""
    init_state = create_init_algorithm_state(minimal_parameters)

    assert isinstance(init_state(), AlgorithmState)


def test_simple_update_algorithm_state(
    minimal_state, minimal_parameters, simple_population
) -> None:
    """Test simple update via designated function"""

    update_state = create_update_algorithm_state(minimal_parameters)

    new_state = update_state(
        original_state=minimal_state, population=simple_population
    )

    assert isinstance(new_state, AlgorithmState)
    assert new_state.generation == minimal_state.generation + 1


@pytest.mark.parametrize("offset", [-1, 1])
def test_update_wrong_population_size(
    minimal_state, minimal_parameters, update_state, offset
) -> None:
    """Test update with wrong population size"""

    population = np.zeros(shape=(minimal_parameters.pop_size + offset,))
    with pytest.raises(ValueError):
        update_state(original_state=minimal_state, population=population)


@pytest.mark.parametrize("offset", [-1, 1])
def test_update_wrong_individual_dimension(
    minimal_state, minimal_parameters, update_state, offset
) -> None:
    """Test update with wrong individual dimension"""

    population = np.zeros(
        shape=(
            minimal_parameters.pop_size,
            minimal_parameters.dimension + offset,
        )
    )
    with pytest.raises(ValueError):
        update_state(
            original_state=minimal_state,
            population=population,
        )


def test_calculate_new_mean(
    minimal_parameters, minimal_state, zeros_population
) -> None:
    """Test simple call of calculate_new_mean function."""

    population = zeros_population
    population[0, 0] = 1.00
    population[1, 0] = 1.00
    population[4, 0] = 0.5
    population[2, 3] = 3.00

    new_mean = calculate_new_mean(
        minimal_parameters, minimal_state, zeros_population
    )

    assert np.isclose(new_mean[0], 0.73978053, atol=1e-3)


def test_rank_one_update_simple(minimal_parameters) -> None:
    """Test simple call of rank-1-update function."""

    A = np.eye(minimal_parameters.dimension)
    beta = 1.00
    v = np.array([1.00] * minimal_parameters.dimension)

    assert rank_one_update(A, beta, v).shape == A.shape


def test_update_state(minimal_state, update_state, simple_population) -> None:
    """Test update of the mean AlgorithmState."""

    new_state = update_state(
        original_state=minimal_state, population=simple_population
    )

    assert isinstance(new_state, AlgorithmState)


def test_simple_sample(minimal_parameters, simple_population) -> None:
    """Test simple sampling of a population."""

    assert simple_population.shape == (
        minimal_parameters.pop_size,
        minimal_parameters.dimension,
    )


def test_sample_and_evaluate(
    minimal_parameters, minimal_state, loss_results
) -> None:
    """Test combined sampling and loss evaluation."""

    sample_func = create_sample_from_state(minimal_parameters)
    eval_func = transposed_rosen
    sample_and_evaluate = create_sample_and_evaluate(sample_func, eval_func)

    population, state, loss = sample_and_evaluate(minimal_state)

    assert population.shape == (10, 8)
    assert np.all(
        np.isclose(
            loss,
            loss_results,
            atol=1e-5,
        )
    )

    assert state != minimal_state


def test_sample_and_sequential_evaluate(
    minimal_parameters, minimal_state, loss_results
) -> None:
    """Test combined sampling and loss evaluation."""

    sample_func = create_sample_from_state(minimal_parameters)
    eval_func = transposed_rosen
    sample_and_evaluate = create_sample_and_sequential_evaluate(
        sample_func, eval_func
    )

    population, state, loss = sample_and_evaluate(minimal_state)

    assert population.shape == (10, 8)
    assert np.all(
        np.isclose(
            loss,
            loss_results,
            atol=1e-5,
        )
    )

    assert state != minimal_state


def test_resample(minimal_parameters, minimal_state) -> None:
    """Test resampling"""
    sample_individuals = create_sample_from_state(minimal_parameters)

    first_sample, first_state = sample_individuals(minimal_state, n_samples=10)

    second_sample, second_state = sample_individuals(first_state, n_samples=10)

    full_sample, _ = sample_individuals(minimal_state, n_samples=20)

    assert first_state != minimal_state
    assert first_state != second_state

    assert np.any(first_sample[0] != second_sample[0])
    assert np.all(first_sample[0] == full_sample[0])
    assert np.all(second_sample[0] == full_sample[10])


def test_reg_rank_one_update(shared_datadir) -> None:
    """Regression test of rank_one_update()."""

    handler = CMAFileHandler(label="reg_test", target_dir=shared_datadir)
    parameters, state, _, _ = handler.load_evolution()

    A = np.load(shared_datadir / "updated_cholesky.npy")
    A_ = rank_one_update(
        state.cholesky_factor,
        parameters.c_1,
        state.p_c,
    )

    assert np.all(A_ == A)


def test_reg_update_state(shared_datadir) -> None:
    """Regression test of update_state()."""

    handler = CMAFileHandler(label="reg_test", target_dir=shared_datadir)
    parameters, _, _, _ = handler.load_evolution()

    gen1 = handler.load_generation(1)
    gen2 = handler.load_generation(2)

    update_state = create_update_algorithm_state(parameters)
    sample_individuals = create_sample_from_state(parameters)
    evaluate_loss = create_sphere_function()
    sample_with_evaluate = create_sample_and_sequential_evaluate(
        sample_individuals, evaluate_loss
    )

    generation, state, loss = sample_with_evaluate(gen1[0])
    idx = np.argsort(loss)
    state = update_state(gen1[0], generation[idx])

    assert np.all(state.cholesky_factor == gen2[0].cholesky_factor)


@pytest.mark.parametrize("workers", [1, 2, 3])
def test_sample_and_run(
    minimal_parameters, minimal_state, loss_results, workers
) -> None:
    """Test combined sampling and loss evaluation with Runner."""

    sample_func = create_sample_from_state(minimal_parameters)
    eval_func = lambda x: (transposed_rosen(x), {})
    runner = FunctionRunner(eval_func, workers)
    sample_and_run = create_sample_and_run(sample_func, runner)

    population_dict, state, loss_dict, _ = sample_and_run(
        minimal_state, minimal_parameters.pop_size
    )
    population = []
    loss = []
    for key, val in population_dict.items():
        try:
            loss.append(loss_dict[key])
            population.append(val)
        except KeyError:
            pass
    population = np.asarray(population)
    loss = np.asarray(loss)

    assert population.shape == (10, 8)
    loss.sort()
    loss_results.sort()
    assert np.all(
        np.isclose(
            loss,
            loss_results,
            atol=1e-5,
        )
    )

    assert state != minimal_state


@pytest.mark.parametrize("workers", [1, 2, 3])
def test_sample_and_run_with_failures(
    minimal_parameters, minimal_state, workers
) -> None:
    """Test combined (re-)sampling and loss evaluation with Runner."""

    rng = np.random.default_rng()
    rng.bit_generator.state = minimal_state.random_state
    sample_func = create_sample_from_state(minimal_parameters)
    runner = FunctionRunner(
        lambda x: (unreliable_rosen_internal_rng(x, threshold=0.5), {}),
        workers,
    )
    sample_and_run = create_sample_and_run(sample_func, runner)

    population_dict, state, loss_dict, _ = sample_and_run(
        minimal_state, 10, n_attempts=5000
    )
    population = []
    loss = []
    for key, val in population_dict.items():
        try:
            loss.append(loss_dict[key])
            population.append(val)
        except KeyError:
            pass
    population = np.asarray(population)
    loss = np.asarray(loss)

    assert population.shape == (10, 8)
    assert state != minimal_state


@pytest.mark.parametrize("workers", [1, 2, 3])
def test_sample_and_run_with_attempts(
    minimal_parameters, minimal_state, workers
) -> None:
    """Test resampling attempt limit."""

    rng = np.random.default_rng()
    rng.bit_generator.state = minimal_state.random_state
    sample_func = create_sample_from_state(minimal_parameters)
    runner = FunctionRunner(
        lambda x: (unreliable_rosen(x, rng.uniform() > 0.1), {}), workers
    )
    sample_and_run = create_sample_and_run(sample_func, runner)

    with pytest.raises(OverflowError):
        _, _, _, _ = sample_and_run(minimal_state, 10, n_attempts=11)


@pytest.mark.parametrize("workers", [1, 2, 3])
def test_sample_and_run_return_failures(
    minimal_parameters, minimal_state, workers
) -> None:
    """Test return of failed evaluations."""

    rng = np.random.default_rng()
    rng.bit_generator.state = minimal_state.random_state
    sample_func = create_sample_from_state(minimal_parameters)
    runner = FunctionRunner(
        lambda x: (unreliable_rosen_internal_rng(x, threshold=0.75), {}),
        workers,
    )
    sample_and_run = create_sample_and_run(sample_func, runner)

    population_dict, state, loss_dict, _, failed_dict = sample_and_run(
        minimal_state, 20, return_failures=True
    )
    population = []
    loss = []
    for key, val in population_dict.items():
        try:
            loss.append(loss_dict[key])
            population.append(val)
        except KeyError:
            pass
    population = np.asarray(population)
    loss = np.asarray(loss)

    assert population.shape == (20, 8)
    assert state != minimal_state

    failed_attempt = failed_dict[list(failed_dict.keys())[0]]

    assert isinstance(failed_attempt[0], ArithmeticError)

    for failed in failed_dict.values():
        assert failed[0].args[0]


def test_resample_and_evaluate(
    minimal_parameters, minimal_state, loss_results
) -> None:
    """Test combined resampling and loss evaluation.

    Without failures -> no resampling."""

    sample_func = create_sample_from_state(minimal_parameters)
    evaluate_batch = transposed_rosen_with_info_and_inputs
    rng = np.random.default_rng()
    rng.bit_generator.state = minimal_state.random_state
    evaluate_single = transposed_rosen_with_info

    resample_and_evaluate = create_resample_and_evaluate(
        sample_func, evaluate_batch, evaluate_single
    )

    population, state, loss, _ = resample_and_evaluate(
        minimal_state, n_samples=minimal_parameters.pop_size
    )

    assert population.shape == (10, 8)
    assert loss.shape == (10, 1)
    assert state != minimal_state


def test_resample_and_evaluate_with_failures(
    minimal_parameters, minimal_state
) -> None:
    """Test combined resampling and loss evaluation.

    With failures -> resampling."""

    sample_func = create_sample_from_state(minimal_parameters)
    evaluate_batch = pretend_to_fail_rosen_with_info
    rng = np.random.default_rng()
    rng.bit_generator.state = minimal_state.random_state
    evaluate_single = lambda x: single_unreliable_rosen_with_info(
        x, rng.uniform() > 0.9
    )

    resample_and_evaluate = create_resample_and_evaluate(
        sample_func, evaluate_batch, evaluate_single
    )

    population, state, loss, _ = resample_and_evaluate(
        minimal_state, n_samples=minimal_parameters.pop_size
    )

    assert population.shape == (10, 8)
    assert loss.shape == (10, 1)
    assert state != minimal_state


def test_resample_and_evaluate_with_attempts(
    minimal_parameters, minimal_state
) -> None:
    """Test combined resampling and loss evaluation.

    With failures -> resampling."""

    sample_func = create_sample_from_state(minimal_parameters)
    evaluate_batch = pretend_to_fail_rosen_with_info
    rng = np.random.default_rng()
    rng.bit_generator.state = minimal_state.random_state
    evaluate_single = lambda x: single_unreliable_rosen_with_info(
        x, rng.uniform() > 0.9
    )

    resample_and_evaluate = create_resample_and_evaluate(
        sample_func, evaluate_batch, evaluate_single
    )

    with pytest.raises(OverflowError):
        resample_and_evaluate(
            minimal_state, n_samples=minimal_parameters.pop_size, n_attempts=11
        )


def test_resample_and_evaluate_return_failures(
    minimal_parameters, minimal_state
) -> None:
    """Test combined resampling and loss evaluation.

    With failures -> resampling."""

    sample_func = create_sample_from_state(minimal_parameters)
    evaluate_batch = pretend_to_fail_rosen_with_info
    rng = np.random.default_rng()
    rng.bit_generator.state = minimal_state.random_state
    evaluate_single = lambda x: single_unreliable_rosen_with_info(
        x, rng.uniform() > 0.9
    )

    resample_and_evaluate = create_resample_and_evaluate(
        sample_func, evaluate_batch, evaluate_single
    )

    population, state, loss, _, failed = resample_and_evaluate(
        minimal_state,
        n_samples=minimal_parameters.pop_size,
        return_failures=True,
    )

    assert population.shape == (10, 8)
    assert loss.shape == (10, 1)
    assert state != minimal_state

    for fail in failed:
        assert isinstance(fail["exception"], ArithmeticError)
        assert fail["exception"].args[0]
