"""This module contains auxiliary test functions as defined in table 1 of

    [1] O. Krause, D. R. Arbonès, C. Igel, "CMA-ES with Optimal Covariance
           Update and Storage Complexity", part of [2], 2016.
    [2] D. Lee, M. Sugiyama, U. Luxburg, I. Guyon, R. Garnett, "Advances
        in Neural Information Processing Systems 29, 2016.
"""

from typing import Callable

import numpy as np
import numpy.typing as npt

__all__ = [
    "create_sphere_function",
    "create_rosenbrock_function",
    "create_discus_function",
    "create_cigar_function",
    "create_ellipsoid_function",
    "create_diffpowers_function",
    "create_ackley_function",
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

        return np.linalg.norm(vec) ** 2

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

        return 1e-6 * vec[0] ** 2 + np.sum(vec[1:] ** 2)

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


def create_diffpowers_function() -> Callable:
    """Different Powers test function.

    Args:

    Returns:
        The Diffpowers test function.
    """

    def diffpowers(vec: npt.ArrayLike) -> float:
        """Different Powers test function as given in Table 1 in Ref [1].

        Args:

        Returns:
            Function value at vec.
        """

        res = 0.0
        for i in range(vec.shape[0]):
            res += np.abs(vec[i]) ** (2.0 + 10.0 * i / (vec.shape[0] - 1.0))
        return res

    return diffpowers


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
        c = 2.0 * np.pi

        return (
            -a * np.exp(-b * np.sqrt(np.mean(vec**2)))
            - np.exp(np.mean(np.cos(c * vec)))
            + a
            + np.exp(1.0)
        )

    return ackley
