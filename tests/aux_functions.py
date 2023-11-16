"""This module contains auxiliary test functions as defined in table 1 of

    [1] O. Krause, D. R. Arbonès, C. Igel, "CMA-ES with Optimal Covariance
           Update and Storage Complexity", part of [2], 2016.
    [2] D. Lee, M. Sugiyama, U. Luxburg, I. Guyon, R. Garnett, "Advances
        in Neural Information Processing Systems 29, 2016.
"""

from typing import Callable, Tuple

import numpy as np
import numpy.typing as npt
from scipy.optimize import rosen

__all__ = [
    "transposed_rosen",
    "transposed_rosen_with_info",
    "transposed_rosen_with_info_and_inputs",
    "unreliable_rosen",
    "single_unreliable_rosen_with_info",
    "pretend_to_fail_rosen_with_info",
]


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


def unreliable_rosen_internal_rng(
    vec: npt.ArrayLike, threshold: float = 0.5
) -> float:
    "Rosenbrock function that fails if the second argument is True"
    rng = np.random.default_rng()
    if rng.uniform() > threshold:
        raise ArithmeticError(True)
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
