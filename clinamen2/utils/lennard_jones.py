"""Implementation of the Lennard Jones potential and related functions."""
import argparse
import pathlib
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from clinamen2.utils.script_functions import cma_parser

FRESH_SAMPLING = "fresh"
CONTINUE_SAMPLING = "continue"


def get_max_span_from_lj_cluster(
    lj_cluster: npt.ArrayLike, verbose=False
) -> float:
    """Evaluate LJ cluster positions and return largest span.

    Args:
        lj_cluster: Positions of the LJ spheres.
    """
    lj_cluster_resh = np.reshape(a=lj_cluster, newshape=(-1, 3))
    max_x_span = abs(lj_cluster_resh[:, 0].min() - lj_cluster_resh[:, 0]).max()
    max_y_span = abs(lj_cluster_resh[:, 1].min() - lj_cluster_resh[:, 1]).max()
    max_z_span = abs(lj_cluster_resh[:, 2].min() - lj_cluster_resh[:, 2]).max()
    if verbose:
        print(f"max x span = {max_x_span}")
        print(f"max y span = {max_y_span}")
        print(f"max z span = {max_z_span}")
    return np.max([max_x_span, max_y_span, max_z_span])


class PositionException(Exception):
    """Exception to be raised for exceeding position threshold."""

    pass


def create_position_filter(
    position_bounds: npt.ArrayLike,
    exception: Exception = BaseException,
) -> Callable:
    """Create function to filter configurations.

    Args:
        position_bounds: Positions components that may not be crossed.
            Shape is (3, 2): 3 components, lower and upper bound for each.
        exception: Exception identifying an issue with the calculation.
    """

    def batch_filter(
        loss: float, additional: list, inputs: Tuple
    ) -> Tuple[float, list]:
        for i, inp in enumerate(inputs):
            for p in range(3):
                positions = np.reshape(a=inp, newshape=(-1, 3))
                if (positions[:, p] < position_bounds[p, 0]).any() or (
                    positions[:, p] > position_bounds[p, 1]
                ).any():
                    additional[i]["exception"] = exception("Out of bounds.")
                    break

        return loss, additional, inputs

    def single_filter(loss: float, additional: list, inputs: Tuple):
        positions = np.reshape(a=inputs, newshape=(-1, 3))
        for p in range(3):
            if (positions[:, p] < position_bounds[p, 0]).any() or (
                positions[:, p] > position_bounds[p, 1]
            ).any():
                additional["exception"] = PositionException("Out of bounds.")
                break

        return loss, additional, inputs

    return single_filter, batch_filter


def create_evaluate_lj_potential(
    n_atoms: int = 38,
    identifier: Optional[str] = None,
    wales_path: pathlib.Path = None,
    n_eval_batch=100,
) -> Callable:
    """Create an LJ evaluation function from a Cambridge database entry

    Load the coordinates of the ground state of an n- atom LJ cluster
    from an entry in

    http://doye.chem.ox.ac.uk/jon/structures/LJ.html

    and return the corresponding LennardJones object and an eval function.

    Args:
        n_atoms: Number of atoms in the cluster.
        identifier: Additional identifier of a specific configuration.
            For example "i" for "38i". Default is None.
        wales_path: Path to the Wales potential data.
        n_eval_batch: Batchsize to be vmapped.

    Returns:
        tuple
            - Coordinates of specified Cluster
            - Evaluation function

    Raises:
        ValueError: If there is a problem with the argument.

    References:

        [1] The Cambridge Cluster Database, D. J. Wales, J. P. K. Doye,
        A. Dullweber, M. P. Hodges, F. Y. Naumkin F. Calvo, J. Hernández-Rojas
        and T. F. Middleton, URL http://www-wales.ch.cam.ac.uk/CCD.html.
    """
    if isinstance(n_atoms, int) or isinstance(n_atoms, str):
        cluster = (
            str(n_atoms) if identifier is None else str(n_atoms) + identifier
        )
        filename = wales_path / cluster
    else:
        raise ValueError("n must be an integer or a string")
    if not filename.exists():
        raise ValueError(
            f"Coordinates for {n_atoms} atoms with identifier"
            f" {identifier} not found in {wales_path}."
        )
    coordinates = np.loadtxt(filename)
    # These coordinates use the sigma = 1 convention -> rescale
    coordinates /= 2 ** (1.0 / 6.0)

    @jax.jit
    def evaluate_lj_potential(positions) -> float:
        """Calculate LJ energy of positions

        Args:
            positions: Flat coordinate vector.

        Returns:
            energy: The energy of the configuration.
        """
        # Compute all distances between pairs without iterating.
        positions = positions.reshape((-1, 3))
        delta = positions[:, jnp.newaxis, :] - positions
        r2 = (delta * delta).sum(axis=2)

        # Take only the upper triangle (combinations of two atoms).
        indices = jnp.triu_indices(r2.shape[0], k=1)
        rm2 = 1.0 / r2[indices]

        # Compute the potental energy recycling some calculations.
        rm6 = rm2 * rm2 * rm2
        return (rm6 * (rm6 - 2.0)).sum(), {}

    vmapped_eval_batch = jax.vmap(evaluate_lj_potential, in_axes=(0))

    def vmapped_eval_wrapper(positions) -> Tuple:
        """Wrapper around the vmapped evaluation.

        Make sure that the additional information is a list of empty
        dictionaries.

        Args:
            positions: LJ sphere positions.

        Returns:
            tuple containing
                - Loss of the configuration.
                - Additional information.
        """

        loss = []
        index_from = 0

        while index_from < positions.shape[0]:
            index_to = min(index_from + n_eval_batch, positions.shape[0])
            positions_batch = positions[index_from:index_to, ...]
            results_batch, _ = vmapped_eval_batch(positions_batch)
            loss.extend(results_batch.tolist())
            index_from = index_to
        loss = jnp.asarray(loss)

        additional = [{}] * loss.shape[0]

        return loss, additional

    return coordinates, evaluate_lj_potential, vmapped_eval_wrapper


def lj_argparse():
    """Argument parser for LJ evolution scripts.

    Returns
        Parsed arguments
    """
    parser = cma_parser()
    parser.add_argument(
        "-a",
        "--atom_count",
        type=int,
        default=38,
        help="Number of atoms in the cluster.",
    )
    parser.add_argument(
        "-i",
        "--identifier",
        type=str,
        default=None,
        help="Identifier for cluster configuration, e.g. 'i'.",
    )
    parser.add_argument(
        "-c",
        "--configuration",
        type=str,
        default="cube",
        help="Shape of initial configuration (cube, sphere, packmol).",
    )
    parser.add_argument(
        "-b",
        "--bounds",
        type=float,
        nargs="+",
        help="Positions bounds as x1 x2 y1 y2 z1 z2",
        default=None,
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Do not write generation data to disk. Only end result.",
    )
    parser.add_argument(
        "-j",
        "--json_output",
        type=str,
        help="JSON file to store results in.",
        default=None,
    )
    parser.add_argument(
        "-w",
        "--wales_path",
        type=str,
        help="Path to Wales potential data.",
        default=None,
    )
    parser.add_argument(
        "-e",
        "--packmol_executable",
        type=str,
        help="Packmol executable with full path.",
        default=None,
    )
    parser.add_argument(
        "--packmol_tolerance",
        type=float,
        help="Packmol tolerance parameter.",
        default=1.0,
    )
    parser.add_argument(
        "--packmol_side_length",
        type=float,
        help="Packmol side_length parameter.",
        default=3.0,
    )
    parser.add_argument(
        "--packmol_seed",
        type=float,
        help="Random seed for packmol. Default is -1",
        default=-1,
    )
    parser.add_argument(
        "--continue_evolution",
        action="store_true",
        help="Set this flag to continue a stopped evolution.",
    )
    parser.add_argument(
        "--generation_checkpoint",
        type=int,
        default=None,
        help="Generation checkpoint to continue from. Use 'last_gen' if None.",
    )
    args, _ = parser.parse_known_args()

    return args
