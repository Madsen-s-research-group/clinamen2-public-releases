"""The core Cholesky CMA-ES implementation.


    References:
        [1] O. Krause, D. R. Arbonès, C. Igel, "CMA-ES with Optimal Covariance
        Update and Storage Complexity", part of [2], 2016.
        
        [2] D. Lee, M. Sugiyama, U. Luxburg, I. Guyon, R. Garnett, "Advances
        in Neural Information Processing Systems 29, 2016.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import scipy as sp
import scipy.stats
from scipy.linalg import solve_triangular

from clinamen2.runner.basic_runner import Runner, WorkerResult


@dataclass(frozen=True)
class AlgorithmParameters:
    """Class for keeping track of the initial, immutable parameters of a run.

    You are unlikely to need to manually instantiate this class. Use
    `init_default_algorithm_parameters` instead, which provides a simpler
    interface covering common use cases and uses recommended defaults
    wherever possible. Directly instantiating this class offers more
    flexibility, but deviating from the literature recommendations for
    parameter defaults without good reason is not encouraged.

    Basic checks on parameters are performed upon direct instantiation,
    but no defaults are used. Every single parameter is therefore required.

    See References for more information on individual parameters.

    Args:
        dimension: The dimensionality of the problem.
        initial_step_size: Global step size at the beginning of a run.
        random_seed: Random seed to start the run with.
        pop_size: The size of the "population", i.e., of each sample of
            random deviates.
        mu: Parent number, equal to the number of positive weights.
            In the standard flavor of CMA-ES (with no negative weights), it is
            the mu best-ranked individuals that contribute to the update of
            the normal distribution.
        weights: The pop_size weights used in the algorithm.
        c_sigma: The learning rate for the conjugate evolution path used for
            step-size control. It must lie in (0, 1).
        d_sigma: The damping term, which must be positive.
        c_m: The learning rate for updating the mean. Generally 1,
            usually <= 1.
        c_c: The learning rate for the evolution path used in the cumulation
            procedure. It must lie in [0, 1].
        c_1: The learning rate for the rank-1 update of the covariance matrix.
            It must lie in [0, 1 - c_mu].
        c_mu: The learning rate for the rank-mu update of the covariance
            matrix. It must lie in [0, 1 - c_1].
    """

    dimension: int
    initial_step_size: float
    random_seed: int
    pop_size: int
    mu: int
    weights: npt.ArrayLike
    c_sigma: float
    d_sigma: float
    c_m: float
    c_c: float
    c_1: float
    c_mu: float

    def __post_init__(self):
        if self.dimension <= 0:
            raise ValueError("'dimension' must be a positive integer.")
        if self.initial_step_size <= 0:
            raise ValueError("'initial_step_size' must be positive.")
        if self.pop_size < 2:
            raise ValueError("'pop_size' must be at least two.")
        if not 0 < self.mu <= self.pop_size:
            raise ValueError("'mu' must be in (0, 'pop_size'].")

        object.__setattr__(self, "weights", np.asarray(self.weights))
        if self.weights.shape != (self.pop_size,):
            raise ValueError(
                "'weights' must be 1-d array_like of length 'pop_size'"
            )
        if np.any(self.weights[: self.mu] < 0):
            raise ValueError("First mu weights must all be nonnegative.")
        if not np.isclose(np.sum(self.weights[: self.mu]), 1):
            raise ValueError("First mu weights must sum to one.")
        if np.any(self.weights[: self.mu - 1] - self.weights[1 : self.mu] < 0):
            raise ValueError(
                "First mu weights must be in non-ascending order."
            )

        object.__setattr__(
            self, "mu_eff", 1 / np.sum(self.weights[: self.mu] ** 2)
        )

        if not 0 < self.c_sigma < 1:
            raise ValueError("'c_sigma' must be in (0, 1).")
        if self.d_sigma <= 0:
            raise ValueError("'d_sigma' must be positive.")
        if not 0 <= self.c_m <= 1:
            raise ValueError("'c_m' must be in [0, 1].")
        if not 0 <= self.c_c <= 1:
            raise ValueError("'c_c' must be in [0, 1].")
        if not 0 <= self.c_1 <= 1 - self.c_mu:
            raise ValueError("'c_1' must be in [0, 1 - c_mu]")
        if not 0 <= self.c_mu <= 1 - self.c_1:
            raise ValueError("'c_mu' must be in [0, 1 - c_1]")


def init_default_algorithm_parameters(
    dimension: int,
    pop_size: int = None,
    alpha_cov: float = 2.0,
    initial_step_size: float = 1.0,
    random_seed: int = 0,
) -> AlgorithmParameters:
    """Initialize CMA-ES parameters to recommended defaults wherever possible.

    This function is the recommended way of instantiating an
    AlgorithmParameters object. It exposes only a minimal set of options to
    the user and computes all other parameters from these using literature
    defaults. The only strictly required argument is the dimension.

    Weights for ranks > mu are initialized to zero, i.e., no negative
    weights as in Ref. [1] are used.

    Args:
        dimension: The dimensionality of the problem, i.e., number of degrees
            of freedom.
        pop_size: The number of samples drawn from the Gaussian at each
            generation. If not given, it is calculated from the dimension
            according to `pop_size =  4 + floor(3 * log(dimension))`.
            Using smaller values than this default is not recommended.
        alpha_cov: A parameter that enters into the calculation of defaults for
            the learning rates used in covariance matrix update.
        initial_step_size: Global step size at the beginning of a run.
        random_seed: Random seed to start the run with.

    Returns:
        An initialized AlgorithmParameters object.
    """
    # Population size and weights
    if pop_size is None:
        lamb = 4 + int(np.floor(3 * np.log(dimension)))
    else:
        lamb = pop_size
    mu = int(np.floor(lamb / 2))
    weights_prime = np.log((lamb + 1) / 2) - np.log(np.arange(1, lamb + 1))
    weights = np.zeros(lamb)
    weights[:mu] = weights_prime[:mu] / np.sum(weights_prime[:mu])
    mu_eff = 1 / np.sum(weights[:mu] ** 2)

    # Step-size control
    c_sigma = (mu_eff + 2) / (dimension + mu_eff + 5)
    d_sigma = (
        1
        + 2 * np.max([0, np.sqrt((mu_eff - 1) / (dimension + 1)) - 1])
        + c_sigma
    )

    # Covariance matrix adaptation
    c_c = (4 + mu_eff / dimension) / (dimension + 4 + 2 * mu_eff / dimension)
    c_1 = alpha_cov / ((dimension + 1.3) ** 2 + mu_eff)
    num = mu_eff - 2 + 1 / mu_eff
    den = (dimension + 2) ** 2 + alpha_cov * mu_eff / 2
    c_mu = np.min([1 - c_1, alpha_cov * num / den])

    return AlgorithmParameters(
        dimension=dimension,
        initial_step_size=initial_step_size,
        random_seed=random_seed,
        pop_size=lamb,
        mu=mu,
        weights=weights,
        c_sigma=c_sigma,
        d_sigma=d_sigma,
        c_m=1.0,
        c_c=c_c,
        c_1=c_1,
        c_mu=c_mu,
    )


@dataclass(frozen=True)
class AlgorithmState:
    """Class for keeping track of the state of the run.

    Includes the CMA-ES data (e.g. ``mean``) and keeps track of the random
    state.

    Args:
        random_state: Dictionary representing the state of the random generator
            numpy.random.default_rng().
        mean: Mean of the Gaussian.
        cholesky_factor: Cholesky factor of the covariance matrix.
        step_size: Global standard deviation of the Gaussian.
        generation: The generation number.
        p_sigma: Evolution path of step size.
        p_c: Evolution path of the covariance.
    """

    random_state: dict
    mean: npt.ArrayLike
    cholesky_factor: npt.ArrayLike
    step_size: float
    generation: int
    p_sigma: npt.ArrayLike
    p_c: npt.ArrayLike

    def __post_init__(self):
        pass


def create_init_algorithm_state(parameters: AlgorithmParameters) -> Callable:
    """Create function that initializes an AlgorithmState.

    Args:
        parameters: Initial, immutable parameters of the CMA-ES run.

    Returns:
        A function for initializing an AlgorithmState.
    """

    def init_algorithm_state(
        mean: Optional[npt.ArrayLike] = None,
        cholesky_factor: Optional[npt.ArrayLike] = None,
        generation: Optional[int] = None,
        p_sigma: Optional[npt.ArrayLike] = None,
        p_c: Optional[npt.ArrayLike] = None,
    ) -> AlgorithmState:
        """Function for creating a new AlgorithmState. Performs all checks.

        Args:
            mean: Mean of the Gaussian.
                Default is zeros of shape (parameters.dimension, ).
            cholesky_factor: Cholesky factor of the covariance matrix.
                Default is the identity matrix of shape
                (parameters.dimension, parameters.dimension)
            p_sigma: Evolution path of step size.
                Default is zeros with shape of the mean.
            p_cov: Evolution path of the covariance.
                Default is zeros with shape of the mean.
            Returns:
                An AlgorithmState with default settings.
        """
        rng = np.random.default_rng(seed=parameters.random_seed)
        random_state = rng.bit_generator.__getstate__()

        if mean is None:
            mean = np.zeros(parameters.dimension)
        else:
            if mean.shape != (parameters.dimension,):
                raise AttributeError(
                    f"The mean vector must have shape "
                    f"{(parameters.dimension,)}"
                )

        if cholesky_factor is None:
            cholesky_factor = np.eye(parameters.dimension)
        else:
            if cholesky_factor.shape != (
                parameters.dimension,
                parameters.dimension,
            ):
                raise AttributeError(
                    f"The cholesky_factor must have shape "
                    f"{(parameters.dimension, parameters.dimension)}"
                )

        step_size = parameters.initial_step_size

        if generation is None:
            generation = 0
        else:
            if not isinstance(generation, int):
                raise TypeError(
                    f"The generation must be an integer >= 0, "
                    f"got {generation}."
                )
            if generation < 0:
                raise ValueError(
                    f"The generation must be an integer >= 0, got {generation}."
                )

        if p_sigma is None:
            p_sigma = np.zeros_like(mean)
        else:
            if p_sigma.shape != mean.shape:
                raise AttributeError(
                    f"The p_sigma vector must have shape "
                    f"{np.zeros_like(mean).shape}"
                )

        if p_c is None:
            p_c = np.zeros_like(mean)
        else:
            if p_c.shape != mean.shape:
                raise AttributeError(
                    f"The p_cov vector must have shape "
                    f"{np.zeros_like(mean).shape}"
                )

        return AlgorithmState(
            random_state=random_state,
            step_size=step_size,
            mean=mean,
            cholesky_factor=cholesky_factor,
            generation=generation,
            p_sigma=p_sigma,
            p_c=p_c,
        )

    if not isinstance(parameters, AlgorithmParameters):
        raise TypeError("Parameters must be of type AlgorithmParameters.")

    return init_algorithm_state


def create_sample_and_evaluate(
    sample_individuals: Callable,
    evaluate_loss: Callable,
    input_pipeline: Callable = lambda x: x,
) -> Callable:
    """Create function that samples a population and evaluates its loss.

    Args:
        sample_individuals: Function that samples a number of individuals from
            a state.
        evaluate_loss: Function that returns the loss of one individual or a
            population of individuals.
        input_pipeline: Transform dof such that evaluate_loss can handle the
            data. Default is identity.

    Returns:
        A function to sample and evaluate a population from a state.
    """

    def sample_and_evaluate(
        state: AlgorithmState,
    ) -> Tuple[npt.ArrayLike, AlgorithmState, npt.ArrayLike]:
        """Function for sampling a population from a state and evaluating
           the loss.

        Args:
            state: State of the previous CMA step.

        Returns:
            tuple containing

            - A population of individuals sampled from the AlgorithmState.
            - The new AlgorithmState.
            - The loss of all individuals.
        """

        population, state = sample_individuals(state)

        loss = evaluate_loss(input_pipeline(population))

        return population, state, loss

    return sample_and_evaluate


def create_sample_and_sequential_evaluate(
    sample_individuals: Callable,
    evaluate_loss: Callable,
    input_pipeline: Callable = lambda x: x,
) -> Callable:
    """Create function that samples a population and evaluates its loss.

    Args:
        sample_individuals: Function that samples a number of individuals from
            a state.
        evaluate_loss: Function that returns the loss of one individual or a
            population of individuals.
        input_pipeline: Transform dof such that evaluate_loss can handle the
            data. Default is identity.

    Returns:
        A function to sample and evaluate a population from a state.
    """

    def sample_and_evaluate(
        state: AlgorithmState,
    ) -> Tuple[npt.ArrayLike, AlgorithmState, npt.ArrayLike]:
        """Function for sampling a population from a state and evaluating
           the loss.

        Args:
            state: State of the previous CMA step.

            Returns:
                tuple containing

                - A population of individuals sampled from the AlgorithmState.
                - The new AlgorithmState.
                - The loss of all individuals.
        """

        population, state = sample_individuals(state)
        loss = []
        for individual in population:
            loss.append(evaluate_loss(input_pipeline(individual)))
        loss = np.array(loss)

        return population, state, loss

    return sample_and_evaluate


def create_resample_and_evaluate(
    sample_individuals: Callable,
    evaluate_batch: Callable,
    evaluate_single: Callable,
    input_pipeline_batch: Callable = lambda x: x,
    input_pipeline_single: Callable = lambda x: x,
) -> Callable:
    """Create function that samples a population and evaluates its loss.

    Samples that are rejected, i.e., they come with an exception are resampled.

    Args:
        sample_individuals: Function that samples a number of individuals from
            a state.
        evaluate_batch: Function that returns a tuple containing the loss of a
            batch of individuals and additional information in a dictionary
            (at least exception if applicable).
        evaluate_single: Function that returns a tuple containing the loss of
            an individual and additional information in a dictionary (at least
            exception if applicable).
        input_pipeline_batch: Transform dof such that evaluate_loss can handle
            the data. Default is identity.
        input_pipeline_single: Transform dof such that evaluate_loss can handle
            the data. Default is identity.

    Returns:
        A function to sample (with resampling) and evaluate a population from a
        state.
    """

    def resample_and_evaluate(
        state: AlgorithmState,
        n_samples: int,
        n_attempts: Optional[int] = int(1e6),
        return_failures: bool = False,
    ) -> Tuple[npt.ArrayLike, AlgorithmState, npt.ArrayLike]:
        """Function for sampling a population from a state and evaluating
           the loss.

        Args:
            state: State of the previous CMA step.
            n_samples: Number of successfully evaluated individuals to be
                returned.
            n_attempts: Maximum number of attempts to reach n_samples.
                Default is 1e6 to avoid infinite loops.
            return_failures: If set to True, individuals for which the
                evaluation failed are returned in a list of dictionaries.

        Returns:
            tuple containing

            - A population of individuals sampled from the AlgorithmState.
            - The new AlgorithmState.
            - An array containing the loss of all passing individuals.
            - Additional information returned by the evaluation function in a
                list of dictionaries, e.g. uncertainty or forces if applicable.
            - Optional: A list of dictionaries containing individuals sampled
                from the AlgorithmState where the evaluation failed. Includes
                at least the associated exception.
        """

        population = []
        loss = []
        information = []
        failed = []
        returned_population, state = sample_individuals(state, n_samples)
        returned_loss, returned_information, _ = evaluate_batch(
            input_pipeline_batch(returned_population)
        )

        # sample and batch evaluate the required samples
        attempts = n_samples
        for i, info in enumerate(returned_information):
            if "exception" in info.keys():
                failed.append(
                    {
                        "individual": returned_population[i],
                        "loss": returned_loss[i],
                        **info,
                    }
                )
            else:
                population.append(returned_population[i])
                loss.append(returned_loss[i])
                information.append(info)

        # resample and single evaluate individuals
        while attempts <= n_attempts and len(population) < n_samples:
            attempts += 1
            resampled_population, state = sample_individuals(
                state, n_samples=1
            )
            resampled_loss, resampled_information, _ = evaluate_single(
                input_pipeline_single(resampled_population)
            )
            if "exception" in resampled_information.keys():
                failed.append(
                    {
                        "individual": resampled_population[0],
                        "loss": resampled_loss,
                        **resampled_information,
                    }
                )
            else:
                population.append(resampled_population[0])
                loss.append(resampled_loss)
                information.append(resampled_information)

        if len(population) < n_samples:
            raise OverflowError(
                f"Evaluation attempt limit of {n_attempts} reached "
            )

        if return_failures:
            return (
                np.asarray(population),
                state,
                np.asarray(loss),
                information,
                failed,
            )
        else:
            return (
                np.asarray(population),
                state,
                np.asarray(loss),
                information,
            )

    return resample_and_evaluate


def create_sample_and_run(
    sample_individuals: Callable,
    runner: Runner,
) -> Callable:
    """Create function that samples a population and evaluates its loss.

    Utilizes an instance of Runner. With resampling for failures.

    Args:
        sample_individuals: Function that samples a number of individuals from
            a state.
        runner: Function that submits all individuals to a Runner and pops
            the results.

    Returns:
        A function to sample and evaluate a population from a state.
    """

    def sample_and_run(
        state: AlgorithmState,
        n_samples: int,
        n_attempts: Optional[int] = int(1e6),
        return_failures: bool = False,
    ) -> Union[
        Tuple[npt.ArrayLike, AlgorithmState, npt.ArrayLike],
        Tuple[npt.ArrayLike, AlgorithmState, npt.ArrayLike, npt.ArrayLike],
    ]:
        """Function for sampling a population from a state and evaluating
           the loss.

        Args:
            state: State of the previous CMA step.
            n_samples: Number of successfully evaluated individuals to be
                returned.
            n_attempts: Maximum number of attempts to reach n_samples.
                Default is 1e6 to avoid infinite loops.
            return_failures: If set to True, individuals for which evaluation
                failed are returned in an additional dictionary.

        Returns:
            tuple containing

            - A dictionary of individuals sampled from the AlgorithmState
                that were successfully evaluated.
            - The new AlgorithmState.
            - A dictionary containing the loss of all individuals.
            - Additional information returned by the evaluation function in a
                list of dictionaries, e.g. atoms object if applicable.
            - Optional: A dictionary containing individuals sampled from
                the AlgorithmState where the evaluation failed.
                Includes the associated exception.
        """

        sampled_dict = {}
        evaluated_dict = {}
        failed_dict = {}
        loss_dict = {}
        information_dict = {}

        population, state = sample_individuals(state, n_samples=n_samples)
        for individual in population:
            key = runner.submit(individual)
            sampled_dict[key] = individual

        for attempt_index, future in enumerate(runner.pop()):
            if attempt_index == n_attempts:
                raise OverflowError(
                    f"Evaluation attempt limit of {n_attempts} reached "
                )
            if future.exception() is None:
                evaluated_dict[future.key] = sampled_dict[future.key]
                result = WorkerResult(*future.result())
                loss_dict[future.key] = result.loss
                information_dict[future.key] = result.information
            else:
                if return_failures:
                    failed_dict[future.key] = (
                        future.exception(),
                        sampled_dict[future.key],
                    )
                resampled, state = sample_individuals(state, n_samples=1)
                key = runner.submit(resampled[0])
                sampled_dict[key] = resampled[0]

        if return_failures:
            return (
                evaluated_dict,
                state,
                loss_dict,
                information_dict,
                failed_dict,
            )
        else:
            return evaluated_dict, state, loss_dict, information_dict

    return sample_and_run


def create_sample_from_state(parameters: AlgorithmParameters) -> Callable:
    """Create function that samples a population.

    Args:
        parameters: Initial, immutable parameters of the CMA-ES run.

    Returns:
        A function to sample invididuals from a state.
    """

    def sample_from_state(
        state: AlgorithmState, n_samples: Optional[int] = None
    ) -> Tuple[npt.ArrayLike, AlgorithmState]:
        """Function for sampling a population from a state

        Args:
            state: State of the previous CMA step.
            n_samples: Number of individuals to be sampled.
                Default is parameters.pop_size.

            Returns:
                tuple containing

                - A population of individuals sampled from the AlgorithmState.
                - The new AlgorithmState.
        """

        if n_samples is None:
            n_samples = parameters.pop_size

        rng = np.random.default_rng()
        rng.bit_generator.state = state.random_state

        # sample from multivariate normal distribution (0, identity)
        base_population = rng.multivariate_normal(
            np.zeros(parameters.dimension),
            np.eye(parameters.dimension),
            n_samples,
        )
        population = []
        for p in range(n_samples):
            individual = (
                state.step_size * (state.cholesky_factor @ base_population[p])
                + state.mean
            )
            population.append(individual)

        new_random_state = rng.bit_generator.__getstate__()
        new_state = AlgorithmState(
            random_state=new_random_state,
            mean=state.mean,
            cholesky_factor=state.cholesky_factor,
            step_size=state.step_size,
            generation=state.generation,
            p_sigma=state.p_sigma,
            p_c=state.p_c,
        )
        return np.array(population), new_state

    return sample_from_state


def calculate_new_mean(
    parameters: AlgorithmParameters,
    original_state: AlgorithmState,
    population: npt.ArrayLike,
) -> npt.ArrayLike:
    """Calculates the new mean of the distribution.

    Args:
        parameters: Initial, immutable parameters of the CMA-ES run.
        original_state: State of the previous CMA step.
        population: Individuals of the current generation.

    Returns:
        The new mean of the distribution.
    """
    centered_population = (
        population - original_state.mean
    ) / original_state.step_size
    weighted_population = (
        parameters.weights[: parameters.mu, np.newaxis]
        * centered_population[: parameters.mu]
    )
    y_w = np.sum(weighted_population[: parameters.mu], axis=0)

    return (
        original_state.mean + parameters.c_m * original_state.step_size * y_w
    )


def rank_one_update(
    A: npt.ArrayLike, beta: float, v: npt.ArrayLike
) -> npt.ArrayLike:
    """Perform the rank-one update of the Cholesky factor as described in [1].

    Args:
        A: Cholesky factor of the covariance matrix.
        beta: Pre-factor (expected: learning rate * weight).
        v: Centered individual.

    Returns:
        Cholesky factor A' of (C + beta*v*v.T).
    """

    alpha = v
    b_factor = 1.0
    n_cols = A.shape[0]

    cols = []
    for j in range(n_cols):
        column_old = A[j:, j]

        left = alpha[0]
        top_left_old = column_old[0]
        top_left_new = np.sqrt(
            top_left_old**2 + (beta / b_factor) * left**2
        )

        gamma = top_left_old**2 * b_factor + beta * left**2

        alpha = alpha[1:] - (left / top_left_old) * column_old[1:]

        new_col = np.concatenate(
            [
                np.zeros(j),
                [top_left_new],
                (top_left_new / top_left_old) * column_old[1:]
                + (top_left_new * beta * left / gamma) * alpha,
            ]
        )
        cols.append(new_col)

        b_factor += beta * left**2 / top_left_old**2

    return np.asarray(cols).T


def create_update_algorithm_state(parameters: AlgorithmParameters) -> Callable:
    """Create function that creates an updated AlgorithmState.

    Args:
        parameters: Initial, immutable parameters of the CMA-ES run.

    Returns:
        A function for getting an updated AlgorithmState.
    """

    def update_algorithm_state(
        original_state: AlgorithmState,
        population: npt.ArrayLike,
    ) -> AlgorithmState:
        """Function for creating an updated AlgorithmState

        Args:
            original_state: The original AlgorithmState to calculate the new
                one from.
            population: An array of all individuals of the original population.

        Returns:
            A new AlgorithmState.
        """

        # check parameters for consistency
        if population.shape[0] != parameters.pop_size:
            raise ValueError(
                f"Size of population does not match expected population "
                f"size of {parameters.pop_size}."
            )
        if population.shape[1] != parameters.dimension:
            raise ValueError(
                f"Dimension {population.shape[1]} of individuals does not "
                f"match expected dimension of {parameters.dimension}."
            )

        mean = calculate_new_mean(parameters, original_state, population)

        p_c = (1.0 - parameters.c_c) * original_state.p_c + np.sqrt(
            parameters.c_c * (2 - parameters.c_c) * parameters.mu_eff
        ) * (mean - original_state.mean) / original_state.step_size

        cholesky_factor = (
            np.sqrt(1.0 - parameters.c_1 - parameters.c_mu)
            * original_state.cholesky_factor
        )
        cholesky_factor = rank_one_update(cholesky_factor, parameters.c_1, p_c)

        for i in range(parameters.mu):
            cholesky_factor = rank_one_update(
                cholesky_factor,
                parameters.c_mu * parameters.weights[i],
                (population[i] - original_state.mean)
                / original_state.step_size,
            )

        cholesky_inv = solve_triangular(
            original_state.cholesky_factor,
            np.eye(parameters.dimension),
            lower=True,
        )

        p_sigma = (
            1.0 - parameters.c_sigma
        ) * original_state.p_sigma + np.sqrt(
            parameters.c_sigma * (2.0 - parameters.c_sigma) * parameters.mu_eff
        ) * cholesky_inv @ (
            (mean - original_state.mean) / original_state.step_size
        )

        step_size = original_state.step_size * np.exp(
            parameters.c_sigma
            / parameters.d_sigma
            * (
                np.linalg.norm(p_sigma)
                / sp.stats.chi.mean(parameters.dimension)
                - 1.0
            )
        )

        return AlgorithmState(
            random_state=original_state.random_state,
            mean=mean,
            cholesky_factor=cholesky_factor,
            step_size=step_size,
            generation=original_state.generation + 1,
            p_sigma=p_sigma,
            p_c=p_c,
        )

    return update_algorithm_state
