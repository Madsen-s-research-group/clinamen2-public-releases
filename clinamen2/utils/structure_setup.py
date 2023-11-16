"""Reusable code for ASE structure setup and ASE related functions.

    Meant to offer simple versions of reusable functions and
    a possible starting point / template for more complex applications.
"""
import os
import copy
import shlex
import pathlib
import tempfile
import contextlib
import subprocess
from typing import Callable, NamedTuple, Tuple, Union

import ase.atoms
import jinja2
import numpy as np
import numpy.typing as npt
from ase.build import sort
from ase.io import read
from scipy.linalg import cholesky


class DofToAtoms:
    """Class that creates an atoms object from a compatible CMA-ES sample.

    The degrees of freedom making up the sample are translated into the
    positions of atoms.

    Args:
        template_atoms: Template of an atoms object.
    """

    def __call__(self, dof: npt.ArrayLike) -> ase.atoms.Atoms:
        """Copy the atoms template and take positions from dof.

        Args:
            dof: Degrees of freedom sampled from the CMA-ES.
        """
        atoms = self.template_atoms.copy()
        positions = dof.reshape((-1, 3))
        atoms.positions[: positions.shape[0]] = positions
        atoms.wrap()

        return atoms

    def __init__(self, template_atoms: ase.atoms.Atoms) -> None:
        """Constructor"""

        self.template_atoms = template_atoms.copy()


class DofPipeline(NamedTuple):
    """Functions to be applied to CMA samples in reverse order.

    Args:
        anchor_to_immutable: Combine dof atoms with immutable atoms.
            Default is None, None function will be skipped.
        dof_to_atoms: Construct an atoms object from a CMA-ES sample.
            Default is None, None function will be skipped.
        apply_weights: Dampen or amplify variation of variables in CMA degrees
            of freedem vector. Default is None, None function will be skipped.
    """

    anchor_to_immutable: Callable = None
    dof_to_atoms: Callable = None
    apply_weights: Callable = None


class AnchorToImmutable:
    """Class that appends immutable atoms to an atoms object.

    Args:
        immutable_atoms: Atoms that always have to be added to complete the
            transformation of a CMA-ES sample to an atoms object to perform
            calculations on.
        sort_atoms: If True, ase.build.sort() will be applied to the resulting
            atoms object.
    """

    def __call__(self, atoms: ase.atoms.Atoms) -> ase.atoms.Atoms:
        """Extend the given atoms by the immutable atoms of the class.

        atoms: Atoms object to be extended.
        """
        result = atoms.copy() + self.immutable_atoms.copy()
        result.wrap()

        return sort(result) if self.sort_atoms else result

    def __init__(
        self, immutable_atoms: ase.atoms.Atoms, sort_atoms: bool = False
    ) -> None:
        """Constructor"""

        self.immutable_atoms = immutable_atoms.copy()
        self.sort_atoms = sort_atoms


def prepare_dof_and_pipeline(
    founder: pathlib.Path,
    scaled_center: Tuple[float, float, float] = None,
    radius: float = None,
) -> npt.ArrayLike:
    """Get dof vector and pipeline for reconstruction.

    The atoms that are accessible by the algorithm can be defined by a sphere.

    Args:
        founder: Full path to founder POSCAR.
        scaled_center: Center point of dof sphere, in scaled coordinates.
        radius: Radius of dof sphere.
    """
    founder_atoms = ase.io.read(founder)

    if scaled_center is not None and radius is not None:
        dist = get_distances_from_scaled_position(
            atoms=founder_atoms, scaled_position=scaled_center
        )
        indices_in = np.where(dist <= radius)[0].tolist()  # ignore defect atom!
        indices_out = np.where(dist > radius)[0].tolist()

        # split founder into dof atoms and immutable atoms
        dof_atoms = founder_atoms.copy()[indices_in]
        immutable_atoms = founder_atoms.copy()[indices_out]

        # create pipeline functions and pipeline
        dof_to_atoms = DofToAtoms(template_atoms=dof_atoms)
        embed_in_anchor = AnchorToImmutable(
            immutable_atoms=immutable_atoms,
            sort_atoms=False,
        )
        pipeline = DofPipeline(
            anchor_to_immutable=embed_in_anchor, dof_to_atoms=dof_to_atoms
        )

        # get degrees of freedom from atoms
        dof = dof_atoms.get_positions().flatten()
    else:
        dof_atoms = founder_atoms.copy()
        dof = dof_atoms.get_positions().flatten()
        pipeline = DofPipeline(dof_to_atoms=DofToAtoms(template_atoms=founder_atoms))

    return dof, create_transform_dof(pipeline), dof_atoms


def create_apply_weights(weights: npt.ArrayLike) -> Callable:
    """Closure to create the apply_weights() function.

    Args:
        weights: Vector of floats to be applied to a CMA-ES sample by
            multiplication.

    Returns:
        Function that applies the given weights to a numpy array.
    """

    def apply_weights(dof: npt.ArrayLike) -> npt.ArrayLike:
        """Apply weights passed to the closure to a CMA-ES sample.

        Args:
            dof: Degrees of freedom sampled by the CMA-ES.

        Returns:
            Manipulated array of same shape as the input.
        """

        try:
            result = dof * weights
        except ValueError as exc:
            raise ValueError(
                f"Weights (shape {weights.shape}) and degrees of freedom"
                f" (shape {dof.shape}) need to have the same shape."
            ) from exc

        return result

    return apply_weights


def create_split_atom(sorted_elements: list) -> Callable:
    """Translate ase.atoms.Atoms object to NeuralIL input.

    That is, positions, types and cell.

    Args:
        sorted_elements: Input for symbol_map setup.
    """
    symbol_map = {s: i for i, s in enumerate(sorted_elements)}

    def split_atom(
        atoms: ase.atoms.Atoms,
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        """Get positions, types and cell from atoms object."""

        types = [symbol_map[s] for s in atoms.symbols]
        return (
            np.array(atoms.positions),
            np.array(types),
            np.array(atoms.cell[...]),
        )

    return split_atom


def create_transform_dof(pipeline: NamedTuple) -> Callable:
    """Closure to create a composite function from DofPipeline.

    Args:
        pipeline: Named tuple containing the functions to be applied in
            reverse order.
    """

    def transform_dof(
        dof: npt.ArrayLike,
    ) -> Union[npt.ArrayLike, ase.atoms.Atoms]:
        """Applies the reversed pipeline given to the closure.

        Args:
            dof: Degrees of freedom as sampled by the CMA-ES.
        """
        transformed = copy.deepcopy(dof)

        for fun in pipeline[::-1]:
            transformed = fun(transformed) if fun is not None else transformed

        return transformed

    return transform_dof


class FilterEvalWorkflow(NamedTuple):
    """Evaluation and filter.

    Args:
        filter: Function that filters (marks) results to be omitted.
        evaluate_loss: Function that calculates the loss from input.
    """

    filter: Callable = None
    evaluate_loss: Callable = None


def create_filter_eval_workflow(workflow: FilterEvalWorkflow) -> Callable:
    """Create evaluation workflow with uncertainty.

    Args:
        workflow: Workflow to create closure from.
    """

    def filter_eval_workflow(inputs: Tuple):
        """Apply evaluation pipeline."""

        evaluated = workflow.evaluate_loss(inputs)

        return workflow.filter(*evaluated, inputs)

    return filter_eval_workflow


def place_atoms_random_cube(
    n_atoms: int = 38, side_length: float = 8.0, random_seed: int = 0
):
    """Place atoms randomly within a cube of a given size.

    The cube is always centered on zero.

    Args:
        n_atoms: Number of atoms to be placed.
        side_length: Side length of the cube.
        random_seed: Seed for the random number generator. Default is 0.
    """
    rng = np.random.default_rng(seed=random_seed)
    arr = rng.uniform(
        low=-(side_length * 0.5), high=side_length * 0.5, size=(n_atoms * 3)
    )

    return arr


def place_atoms_random_sphere(
    n_atoms: int = 38, radius: float = 3.5, random_seed: int = 0
):
    """Place atoms randomly within a sphere of a given radius.

    The sphere is always centered at zero.

    Args:
        n_atoms: Number of atoms to be placed.
        radius: Radius of the sphere.
        random_seed: Seed for the random number generator. Default is 0.
    """
    rng = np.random.default_rng(seed=random_seed)
    phi = rng.uniform(low=0.0, high=2.0 * np.pi, size=n_atoms)
    costheta = rng.uniform(low=-1.0, high=1.0, size=n_atoms)
    u = rng.random(size=n_atoms)
    theta = np.arccos(costheta)
    r = radius * np.cbrt(u)

    points = np.asarray(
        [
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta),
        ]
    ).T.flatten()

    return points


# Simple template for an input file telling packmol to put spheres
# in a parallelepiped box.
PACKMOL_INPUT_TEMPLATE = """tolerance {tolerance}
filetype xyz 
output {output_file}
seed {random_seed}

structure {structure_file}
  number {n_atoms}
  inside box 0. 0. 0. {side_length} {side_length} {side_length}
end structure
"""

# Contents of a minimal XYZ structure file for Packmol, containing
# a  single atom.
TRIVIAL_PACKMOL_STRUCT = """1
sphere
H   1.000   1.000   1.00
"""


@contextlib.contextmanager
def dir_context(dir_name: pathlib.Path):
    """Create a context to run code in a different directory.

    Args:
        dir_name: Route to the directory.
    """
    cwd = os.getcwd()
    try:
        os.chdir(dir_name)
        yield
    finally:
        os.chdir(cwd)


def place_atoms_packmol(
    n_atoms: int = 38,
    side_length: float = 5.0,
    tolerance: float = 1.0,
    exec_string: str = None,
    random_seed: int = -1,
):
    """Place atoms within a volume utilizing packmol.

    Packmol is started as subprocess with limited error handling. For a
    detailed description see packmol documentation.
    Input file is created from a jinja2 template.

    Args:
        n_atoms: Number of atoms to be placed.
        side_length: Atoms are placed within a cube of this side length.
        tolerance: Packmol parameter tolerance (interatomic distance).
        exec_string: Packmol executable with full path.
        random_seed: Seed for packmol input. Default is -1, which leads to
            packmol generating a random seed.

    References:
        [1]: L. Martínez, R. Andrade, E. G. Birgin, J. M. Martínez.
        J. Comput. Chem., 30(13):2157-2164, 2009.
    """
    if exec_string is None:
        raise ValueError("'exec_string' may not be ' None'.")

    structure_file = "trivial.xyz"
    input_file = f"box_{n_atoms}.inp"
    output_file = f"box_{n_atoms}.xyz"
    packmol_dict = {
        "n_atoms": n_atoms,
        "side_length": side_length,
        "tolerance": tolerance,
        "random_seed": int(random_seed),
        "structure_file": structure_file,
        "output_file": output_file,
    }
    packmol_input = PACKMOL_INPUT_TEMPLATE.format(**packmol_dict)
    print(packmol_input)
    with tempfile.TemporaryDirectory() as folder_name:
        with dir_context(folder_name):
            with open(structure_file, "w", encoding="UTF-8") as struct_f:
                struct_f.write(TRIVIAL_PACKMOL_STRUCT)
            with open(input_file, "w", encoding="UTF-8") as input_f:
                input_f.write(packmol_input)
            with open(input_file, "r", encoding="UTF-8") as input_f:
                subprocess.run(
                    shlex.split(exec_string),
                    stdin=input_f,
                    check=False,
                )
            atoms = read(output_file)

    return atoms.get_positions()


def get_distances_from_position(
    atoms: ase.atoms.Atoms, position: Tuple[float, float, float]
):
    """Get distances to position in Angstrom.

    Args:
        atoms: Atoms object to calculate distances for.
        position: Position to calculate distance from.
    """

    atoms_copy = atoms.copy()
    atom_at_position = ase.atoms.Atom(symbol="H", position=position)
    atoms_copy.append(atom_at_position)

    # get distance to atom at given position for all atoms
    distances = atoms_copy.get_all_distances(mic=True)[-1]

    return distances[:-1]  # exclude defect atom


def get_distances_from_scaled_position(
    atoms: ase.atoms.Atoms, scaled_position: Tuple[float, float, float]
):
    """Wrapper for get_distances_from_position().

    Args:
        atoms: Atoms object to calculate distances for.
        scaled_position: Position to calculate distance from.
    """
    return get_distances_from_position(
        atoms=atoms, position=scaled_position @ atoms.cell
    )


def bias_covariance_matrix_gauss(
    atoms: ase.atoms.Atoms,
    scaled_position: npt.ArrayLike,
    c_r: float,
    sigma_cov: float,
    dimension: int,
) -> npt.ArrayLike:
    """Return Cholesky factor of biased covariance matrix.

    Use Gaussian for decay.

    Args:
        atoms: Input structure.
        scaled_positions: Positions to compare to (in [0, 1]).
        c_r: Overall bias weight parameter.
        sigma_cov: Gaussian weight parameter.
        dimension: Dimension of the input problem.
    """
    dist = get_distances_from_scaled_position(
        atoms=atoms, scaled_position=scaled_position
    )
    c_r_i = c_r * np.exp(-(dist**2) / (2.0 * sigma_cov**2))
    cov = np.identity(dimension) + np.diag(np.repeat(c_r_i**2, 3))

    return cholesky(cov)


def bias_covariance_matrix_r(
    atoms: ase.atoms.Atoms,
    scaled_position: npt.ArrayLike,
    c_r: float,
    dimension: int,
) -> npt.ArrayLike:
    """Return Cholesky factor of biased covariance matrix.

    Args:
        atoms: Input structure.
        scaled_positions: Positions to compare to (in [0, 1]).
        c_r: Overall bias weight parameter.
        dimension: Dimension of the input problem.

    Ref:
        [1] M. Arrigoni et al., npj Comput. Mater., 2021, 7, 1-13.
    """
    dist = get_distances_from_scaled_position(
        atoms=atoms, scaled_position=scaled_position
    )
    c_r_i = c_r / (1.0 + dist) ** 2
    cov = np.identity(dimension) + np.diag(np.repeat(c_r_i**2, 3))

    return cholesky(cov)
