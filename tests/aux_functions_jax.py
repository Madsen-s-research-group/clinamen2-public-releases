"""This module contains auxiliary test functions as defined in table 1 of

    [1] O. Krause, D. R. Arbonès, C. Igel, "CMA-ES with Optimal Covariance
           Update and Storage Complexity", part of [2], 2016.
    [2] D. Lee, M. Sugiyama, U. Luxburg, I. Guyon, R. Garnett, "Advances
        in Neural Information Processing Systems 29, 2016.
"""

from typing import Callable, Tuple

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from scipy.optimize import rosen

__all__ = [
    "create_sphere_function",
    "create_rosenbrock_function",
    "create_discus_function",
    "create_cigar_function",
    "create_ellipsoid_function",
    "create_ackley_function",
    "transposed_rosen",
    "transposed_rosen_with_info",
    "transposed_rosen_with_info_and_inputs",
    "unreliable_rosen",
    "single_unreliable_rosen_with_info",
    "pretend_to_fail_rosen_with_info",
]


def create_sphere_function() -> Callable:
    """Create sphere test function.

    Args:

    Returns:
        The Sphere test function.
    """

    def sphere(vec: npt.ArrayLike) -> float:
        """Discus test function as given in Table 1 in Ref [1].

        Args:

            Returns:
                Function value at vec.
        """

        return jnp.linalg.norm(vec) ** 2

    return sphere


def create_rosenbrock_function() -> Callable:
    """Create Rosenbrock test function.

    Args:

    Returns:
        The Rosenbrock test function.
    """

    def rosenbrock(vec: npt.ArrayLike) -> float:
        """Rosenbrock test function as given in Table 1 in Ref [1].

        Args:

            Returns:
                Function value at vec.
        """

        res = 0.0
        for i in range(vec.shape[0] - 1):
            res += 100.0 * (vec[i + 1] - vec[i] ** 2) ** 2 + (1 - vec[i]) ** 2
        return res

    return rosenbrock


def create_discus_function() -> Callable:
    """Create discus test function.

    Args:

    Returns:
        The Discus test function.
    """

    def discus(vec: npt.ArrayLike) -> float:
        """Discus test function as given in Table 1 in Ref [1].

        Args:

            Returns:
                Function value at vec.
        """

        return vec[0] ** 2 + np.sum(1e-6 * vec[1:] ** 2)

    return discus


def create_cigar_function() -> Callable:
    """Create cigar test function.

    Args:

    Returns:
        The Cigar test function.
    """

    def cigar(vec: npt.ArrayLike) -> float:
        """Cigar test function as given in Table 1 in Ref [1].

        Args:

            Returns:
                Function value at vec.
        """

        return 1e-6 * vec[0] ** 2 + jnp.sum(vec[1:] ** 2)

    return cigar


def create_ellipsoid_function() -> Callable:
    """Create ellipsoid test function.

    Args:

    Returns:
        The Ellipsoid test function.
    """

    def ellipsoid(vec: npt.ArrayLike) -> float:
        """Ellipsoid test function as given in Table 1 in Ref [1].

        Args:

            Returns:
                Function value at vec.
        """

        res = 0.0
        for i in range(vec.shape[0]):
            res += 10 ** (-6.0 * i / (vec.shape[0] - 1)) * vec[i] ** 2
        return res

    return ellipsoid


def create_ackley_function() -> Callable:
    """Create Ackley test function.

    Args:

    Returns:
        The Ackley test function.
    """

    def ackley(vec: npt.ArrayLike) -> float:
        """Ackley test function.

        Args:

            Returns:
                Function value at vec.
        """

        a = 20.0
        b = 0.2
        c = 2.0 * jnp.pi

        return (
            -a * jnp.exp(-b * jnp.sqrt(jnp.mean(vec**2)))
            - jnp.exp(jnp.mean(jnp.cos(c * vec)))
            + a
            + jnp.exp(1.0)
        )

    return ackley


def transposed_rosen(vec: npt.ArrayLike) -> float:
    """The function expects the transposed shape of what is used here."""
    return rosen(vec.T)


def transposed_rosen_with_info(vec: npt.ArrayLike) -> Tuple[float, dict]:
    """The function expects the transposed shape from what is used here."""
    return rosen(vec.T), {}, vec


def transposed_rosen_with_info_and_inputs(
    vec: npt.ArrayLike,
) -> Tuple[float, dict]:
    """The function expects the transposed shape from what is used here."""
    return rosen(vec.T), {}, vec


def unreliable_rosen(vec: npt.ArrayLike, must_fail: bool = False) -> float:
    "Rosenbrock function that fails if the second argument is True"
    if must_fail:
        raise ArithmeticError(must_fail)
    return rosen(vec)


def single_unreliable_rosen_with_info(
    vec: npt.ArrayLike, must_fail: bool = False
) -> Tuple[float, dict]:
    "Rosenbrock function that fails if the second argument is True"
    if must_fail:
        information = {"exception": ArithmeticError(True)}
    else:
        information = {}

    return transposed_rosen(vec), information, vec


def pretend_to_fail_rosen_with_info(vec: npt.ArrayLike) -> Tuple[float, dict]:
    "Rosenbrock function but every third call 'pretend fails'"
    information = [{"exception": None}] * vec.shape[0]
    result = transposed_rosen(vec)

    for i in range(vec.shape[0]):
        if not (i + 1) % 3:
            information[i]["exception"] = ArithmeticError(True)

    return result, information, {}
