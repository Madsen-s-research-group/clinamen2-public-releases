"""Extension of the structure_setup.py module.

Includes implementations of reflection and rotation operations for the
reduction of the degrees of freedom.
"""

import pathlib
from enum import Enum
from typing import Callable, List, NamedTuple, Tuple

import ase.atoms
import numpy as np
import numpy.typing as npt
from ase.build import sort
from scipy.linalg import cholesky

from clinamen2.utils.structure_setup import (
    AnchorToImmutable,
    create_transform_dof,
    get_distances_from_scaled_position,
)


class Operation(str, Enum):
    ROTATION = "rotation"
    REFLECTION = "reflection"


class Symmetry(str, Enum):
    NONE = "none"
    HORIZONTAL = "horizontal"
    THREEFOLD = "threefold"
    THREEFOLD_HORIZONTAL = "threefold_horizontal"
    SIXFOLD_HORIZONTAL = "sixfold_horizontal"


class DofToAtomsSym:
    """Class that creates an atoms object from a compatible CMA-ES sample.

    The degrees of freedom making up the sample are translated into the
    positions of atoms.

    Args:
        template_atoms: Template of an atoms object.
        symmetry_array: Array that specifies which atoms are placed on
            special symmetry positions such as mirror planes or rotation axes,
            which atom coordinates are degrees of freedom and which are fixed
            during the evolution.
    """

    def __call__(
        self,
        dof: npt.ArrayLike,
    ) -> ase.atoms.Atoms:
        """Copy the atoms template and take positions from dof.

        Args:
            dof: Degrees of freedom sampled from the CMA-ES.
            axes: Tuple of coordinate axes for reference.
        """

        atoms = self.template_atoms.copy()
        dof_list = dof.tolist().copy()

        # Fill every coordinate that is not fixed in every atom that is part of
        # dof with the corresponding value from the dof list
        for atom in self.symmetry_array:
            if atom["dof_atom"]:
                for axis in np.where(np.logical_not(atom["position_fixed"]))[
                    0
                ]:
                    atom["position_coord"][axis] = dof_list.pop(0)

        # Construct new dof by appending all atom coordinates
        dof = self.symmetry_array["position_coord"].flatten()

        # Set the new positions in the atoms object
        positions = dof.reshape((-1, 3))
        atoms.positions[: positions.shape[0]] = positions
        atoms.wrap()

        return atoms

    def __init__(
        self,
        template_atoms: ase.atoms.Atoms,
        symmetry_array: npt.ArrayLike,
    ) -> None:
        """Constructor"""

        self.template_atoms = template_atoms.copy()
        self.symmetry_array = symmetry_array


def bias_covariance_matrix_r_sym(
    atoms: ase.atoms.Atoms,
    scaled_position: npt.ArrayLike,
    c_r: float,
    dimension: int,
    symmetry_array: npt.ArrayLike,
) -> npt.ArrayLike:
    """Return Cholesky factor of biased covariance matrix.

    Args:
        atoms: Input structure.
        scaled_positions: Positions to compare to (in [0, 1]).
        c_r: Overall bias weight parameter.
        dimension: Dimension of the input problem.
        symmetry_array: Array that specifies which atoms are placed on
            special symmetry positions such as mirror planes or rotation axes,
            which atom coordinates are degrees of freedom and which are fixed
            during the evolution.
        axes: Tuple of coordinate axes for reference.

    Ref:
        [1] M. Arrigoni et al., npj Comput. Mater., 2021, 7, 1-13.
    """

    # Collect the indices of all dof atoms in the atom_indices list
    atom_indices = np.where(symmetry_array["dof_atom"])[0]

    # Calculate the distances for all dof atoms
    distances = get_distances_from_scaled_position(
        atoms=atoms[atom_indices],
        scaled_position=scaled_position,
    )

    # Calculate the c_r values for all distances
    c_r_i = c_r / (1.0 + distances) ** 2
    c = np.repeat(c_r_i**2, 3)

    # Collect the indices of all fixed coordinates from the dof atoms. A counter
    # recordes the number of skipped coordiantes by adding the number of axes
    # every time a non-dof atom is passed. This counter is later subtracted
    # since the biased covariance matrix only takes dof atoms into account
    fixed_indices = []
    counter = 0
    for index, atom in enumerate(symmetry_array):
        if not atom["dof_atom"]:
            counter += atom["position_fixed"].shape[0]
        if atom["dof_atom"]:
            for axis in np.where(atom["position_fixed"])[0]:
                fixed_indices.append(
                    atom["position_fixed"].shape[0] * index + axis - counter
                )

    # Remove all fixed coordinates from the c_r values array
    c = np.delete(c, fixed_indices)

    # Construct the biased covariance matrix
    cov = np.identity(dimension) + np.diag(c)

    return cholesky(cov)


class DofPipelineSym(NamedTuple):
    """Functions to be applied to CMA samples in reverse order.

    Args:
        anchor_to_immutable: Combine dof atoms with immutable atoms.
            Default is None, None function will be skipped.
        apply_third_symmetry_operation: Apply the third specified symmetry
            operation on DOF. Default is None, None function will be skipped.
        apply_second_symmetry_operation: Apply the second specified symmetry
            operation on DOF. Default is None, None function will be skipped.
        apply_first_symmetry_operation: Apply the first specified symmetry
            operation on DOF. Default is None, None function will be skipped.
        dof_to_atoms: Construct an atoms object from a CMA-ES sample.
            Default is None, None function will be skipped.
    """

    anchor_to_immutable: Callable = None
    apply_third_symmetry_operation: Callable = None
    apply_second_symmetry_operation: Callable = None
    apply_first_symmetry_operation: Callable = None
    dof_to_atoms: Callable = None


def create_dof_from_atoms(
    dof_atoms: ase.atoms.Atoms,
    symmetry_array: npt.ArrayLike,
) -> Tuple[npt.ArrayLike, dict]:
    """Extract the degrees of freedom from the dof atoms by using the symmetry
    indices.

    Args:
        dof_atoms: Atoms object that containes the dof atoms.
        symmetry_array: Array that specifies which atoms are placed on
            special symmetry positions such as mirror planes or rotation axes,
            which atom coordinates are degrees of freedom and which are fixed
            during the evolution.
        axes: List of coordinate axes for reference.
    """

    # Fill the symmetry indices array with the value of every coordinate
    # of every atom and additionally set them to not fixed
    symmetry_array["position_coord"] = dof_atoms.get_positions().copy()
    symmetry_array["position_fixed"] = False

    # If atoms are placed in a mirror plane, the coordinate in the direction of
    # the plane normal vector is fixed. If atoms are placed on a rotation axis,
    # the other two coordinates not part of the rotation axis are fixed. Only
    # these axes are checked, where at least one value is set to True
    for atom in symmetry_array:
        if atom["plane_atom"].sum() > 0:
            for axis in np.where(atom["plane_atom"])[0]:
                atom["position_fixed"][axis] = True
        if atom["axis_atom"].sum() > 0:
            for axis in np.where(np.logical_not(atom["axis_atom"]))[0]:
                atom["position_fixed"][axis] = True

    # Construct the dof list by appending all coordinates from the dof atoms
    # that are not fixed. Also mark all coordinates as fixed if they are part
    # of a non-dof atom in the symmetry indices array
    dof = []
    for atom in symmetry_array:
        if not atom["dof_atom"]:
            atom["position_fixed"] = True
        if atom["dof_atom"]:
            for axis in np.where(np.logical_not(atom["position_fixed"]))[0]:
                dof.append(atom["position_coord"][axis])

    dof = np.array(dof).flatten()

    return dof, symmetry_array


def split_founder_atoms(
    founder_atoms: ase.atoms.Atoms,
    center_position: npt.ArrayLike,
    radius: float,
) -> Tuple[ase.atoms.Atoms, ase.atoms.Atoms]:
    """Split the founder atoms into dof atoms and immutable atoms.

    Args:
        founder_atoms: Atoms object that containes the founder structure.
        center_position: Center of the degrees of freedom sphere.
        radius: Radius of the degrees of freedom sphere.
        threshold: Small value to be used as lower limit condition to exclude
            the center atom from the dof atoms.
    """

    # Calculate the distances for all founder atoms
    distances = get_distances_from_scaled_position(
        atoms=founder_atoms,
        scaled_position=center_position,
    )

    # Split the founder atoms by including all atoms inside the sphere radius
    # in the dof atoms and all atoms outside in the immutable atoms.
    indices_in = np.where(distances <= radius)[0].tolist()
    indices_out = np.where(distances > radius)[0].tolist()

    dof_atoms = founder_atoms.copy()[indices_in]
    immutable_atoms = founder_atoms.copy()[indices_out]

    return dof_atoms, immutable_atoms


def prepare_dof_for_reflection(
    dof_atoms: ase.atoms.Atoms,
    symmetry_array: npt.ArrayLike,
    plane_position: npt.ArrayLike,
    plane_normal_vector_axis: int,
    plane_angle: float,
    threshold: float,
    additional_condition_vector: npt.ArrayLike = None,
    flip_threshold: float = None,
) -> Tuple[ase.atoms.Atoms, dict]:
    """Modify the symmetric indices array to include all necessary
    information for the reflection operation.

    Args:
        dof_atoms: Atoms object that containes the dof atoms.
        symmetry_array: Array that specifies which atoms are placed on
            special symmetry positions such as mirror planes or rotation axes,
            which atom coordinates are degrees of freedom and which are fixed
            during the evolution.
        plane_position: Position vector to a point in space which is part of
            the mirror plane.
        plane_normal_vector_axis: Axis which is normal to the mirror plane.
        plane_angle: Angle of the mirror plane in relation to the unit cell
            basis vectors.
        threshold: Additional parameter to shift specific plane positions.
        additional_condition_vector: Additional plane normal vector for an
            optional second plane to help define the reflection atoms.
        flip_threshold: Additional parameter to flip/reflect the additional
            condition vector to the other side of the mirror plane in a
            specific angle.
    """

    # Define the plane normal vector dependent on the mirror axis.
    if plane_normal_vector_axis == 0:
        plane_normal_vector = np.array([1, 0, 0])
    elif plane_normal_vector_axis == 1:
        plane_normal_vector = np.array([0, 1, 0])
    elif plane_normal_vector_axis == 2:
        plane_normal_vector = np.array([0, 0, 1])
    else:
        raise ValueError(
            "Currently only plane_normal_vector_axis 0, 1 or 2 implemented!"
        )

    # Construct small vector to shift specific plane positions if necessary.
    eps = np.zeros(3)
    eps[np.where(plane_normal_vector != 0)] = threshold

    # Mark atoms on the other side of the mirror plane as fixed.
    above_plane_indices = get_atom_indices_above_plane(
        atoms_object=dof_atoms,
        plane_position=plane_position + eps,
        plane_normal_vector=plane_normal_vector,
        plane_angle=plane_angle,
    )

    for index in above_plane_indices:
        symmetry_array[index]["dof_atom"] = False

    # Mark atoms on one side of the mirror plane as reflection atoms. These are
    # the atoms which represent the to be reflected atoms before the operation.
    if additional_condition_vector == None:
        reflection_indices = get_atom_indices_above_plane(
            atoms_object=dof_atoms,
            plane_position=plane_position + (-1) * eps,
            plane_normal_vector=(-1) * plane_normal_vector,
            plane_angle=plane_angle,
        )
    else:
        reflection_indices = get_atom_indices_between_planes(
            atoms_object=dof_atoms,
            first_plane_position=plane_position + (-1) * eps,
            first_plane_normal_vector=(-1) * plane_normal_vector,
            second_plane_position=plane_position + (-1) * eps,
            second_plane_normal_vector=np.array(additional_condition_vector),
            plane_angle=plane_angle,
        )

    for index in reflection_indices:
        symmetry_array[index]["before_reflection_atom"][
            plane_normal_vector_axis
        ] = True

    # Mark atoms on other side of the mirror plane as reflected atoms. These are
    # the atoms which represent the then reflected atoms after the operation.
    if additional_condition_vector == None:
        reflected_indices = get_atom_indices_above_plane(
            atoms_object=dof_atoms,
            plane_position=plane_position + eps,
            plane_normal_vector=plane_normal_vector,
            plane_angle=plane_angle,
        )
    else:
        flip = np.ones(3)
        flip[np.where(plane_normal_vector != 0)] = flip_threshold
        flipped_additional_condition_vector = (
            additional_condition_vector * flip
        )

        reflected_indices = get_atom_indices_between_planes(
            atoms_object=dof_atoms,
            first_plane_position=plane_position + eps,
            first_plane_normal_vector=plane_normal_vector,
            second_plane_position=plane_position + (-1) * eps,
            second_plane_normal_vector=flipped_additional_condition_vector,
            plane_angle=plane_angle,
        )

    for index in reflected_indices:
        symmetry_array[index]["after_reflection_atom"][
            plane_normal_vector_axis
        ] = True

    # Mark the atoms positioned inside the mirror plane as fixed.
    plane_indices = get_atom_indices_inside_plane(
        atoms_object=dof_atoms,
        plane_position=plane_position,
        plane_normal_vector=plane_normal_vector,
        plane_angle=plane_angle,
        eps=eps,
    )

    for index in plane_indices:
        symmetry_array[index]["plane_atom"][plane_normal_vector_axis] = True

    return dof_atoms, symmetry_array


def prepare_dof_for_rotation(
    dof_atoms: ase.atoms.Atoms,
    symmetry_array: npt.ArrayLike,
    axis_position: npt.ArrayLike,
    axis_vector_axis: int,
    rotation_angle: float,
    begin_angle: float,
    end_angle: float,
    threshold: float,
    eps_angle: float = -2.0,
) -> Tuple[ase.atoms.Atoms, dict]:
    """Modify the symmetric indices array to include all necessary
    information for the rotation operation.

    Args:
        dof_atoms: Atoms object that containes the dof atoms.
        symmetry_array: Array that specifies which atoms are placed on
            special symmetry positions such as mirror planes or rotation axes,
            which atom coordinates are degrees of freedom and which are fixed
            during the evolution.
        axis_position: Position vector to a point in space which is part of
            the rotation axis.
        axis_vector_axis: Axis which defines the direction of the rotation axis.
        rotation_angle: Angle of the rotation operation. 360 degrees should be
            divisible by this value without a remainder.
        begin_angle: Angle that defines the start of the rotation atoms slice.
        end_angle: Angle that defines the end of the rotation atoms slice.
        threshold: Additional parameter to shift specific plane positions.
        eps_angle: Additional parameter to shift specific angle values.
    """

    # Define the angle vectors dependent on the rotation axis and slice angles.
    begin_normal_vector = angle_vector(axis_vector_axis, begin_angle)
    end_normal_vector = angle_vector(axis_vector_axis, end_angle)

    # Construct small vector to shift specific angle values if necessary.
    eps = np.zeros(3)
    eps[np.where(begin_normal_vector != 0)] = threshold

    # Mark atoms outside one plane of the rotation slice as fixed.
    above_plane_indices = get_atom_indices_above_plane(
        atoms_object=dof_atoms,
        plane_position=axis_position + eps,
        plane_normal_vector=begin_normal_vector,
        plane_angle=begin_angle,
    )

    for index in above_plane_indices:
        symmetry_array[index]["dof_atom"] = False

    # Mark atoms outside the other plane of the rotation slice as fixed.
    above_plane_indices = get_atom_indices_above_plane(
        atoms_object=dof_atoms,
        plane_position=axis_position + eps,
        plane_normal_vector=end_normal_vector,
        plane_angle=begin_angle,
    )

    for index in above_plane_indices:
        symmetry_array[index]["dof_atom"] = False

    # Mark atoms inside of the rotation slice as rotation atoms. These are
    # the atoms which represent the to be rotated atoms before the operation.
    rotation_indices = get_atom_indices_between_planes(
        atoms_object=dof_atoms,
        first_plane_position=axis_position + eps,
        first_plane_normal_vector=(-1) * begin_normal_vector,
        second_plane_position=axis_position + eps,
        second_plane_normal_vector=(-1) * end_normal_vector,
        plane_angle=begin_angle,
    )

    for index in rotation_indices:
        symmetry_array[index]["before_rotation_atom"][axis_vector_axis] = True

    # Mark the atoms positioned on the rotation axis as fixed.
    axis_indices = get_atom_indices_around_axis(
        dof_atoms=dof_atoms,
        axis_position=axis_position,
        axis_vector_axis=axis_vector_axis,
    )

    if axis_indices:
        for index in axis_indices:
            symmetry_array[index]["axis_atom"][axis_vector_axis] = True
            symmetry_array[index]["before_rotation_atom"][
                axis_vector_axis
            ] = False

    # Mark atoms outside of the original rotation slice as rotated atoms,
    # stepwise for each iteration of the rotation operation. These are the
    # atoms which represent the then rotated atoms after the operation.
    for i, angle in enumerate(range(rotation_angle, 360, rotation_angle)):

        b_angle = i * rotation_angle + (i + 1) * end_angle
        begin_normal_vector = angle_vector(axis_vector_axis, b_angle)

        e_angle = (i + 1) * rotation_angle + (i + 2) * end_angle + eps_angle
        end_normal_vector = angle_vector(axis_vector_axis, e_angle)

        rotated_indices = get_atom_indices_between_planes(
            atoms_object=dof_atoms,
            first_plane_position=axis_position + eps,
            first_plane_normal_vector=begin_normal_vector,
            second_plane_position=axis_position + eps,
            second_plane_normal_vector=(-1) * end_normal_vector,
            plane_angle=begin_angle,
        )

        n_rot = int(angle / rotation_angle) - 1
        for index in rotated_indices:
            symmetry_array[index]["after_rotation_atom"][n_rot][
                axis_vector_axis
            ] = True

    return dof_atoms, symmetry_array


def angle_vector(
    axis_vector_axis: int,
    angle: float,
) -> npt.ArrayLike:
    """Construct an angle vector dependent on the rotation axis and the angle.

    Args:
        axis_vector_axis: Axis which defines the direction of the rotation axis.
        angle: Angle of the angle vector.
    """

    # Define the index positions of the sine and cosine function values inside
    # the angle vector dependent on the given rotation axis
    if axis_vector_axis == 0:
        cos_index, sin_index, zero_index = 1, 2, 0
    elif axis_vector_axis == 1:
        cos_index, sin_index, zero_index = 2, 0, 1
    elif axis_vector_axis == 2:
        cos_index, sin_index, zero_index = 0, 1, 2
    else:
        raise ValueError(
            "Currently only axis_vector_axis 0, 1, 2 implemented!"
        )

    # Fill the angle vector with the sine and cosine function values
    angle_vector = np.zeros(3)
    angle_vector[cos_index] = np.cos(np.radians(angle))
    angle_vector[sin_index] = np.sin(np.radians(angle))
    angle_vector[zero_index] = 0.0

    return angle_vector


def get_atom_indices_around_axis(
    dof_atoms: ase.atoms.Atoms,
    axis_position: npt.ArrayLike,
    axis_vector_axis: int,
    threshold: float = 0.1,
) -> List[int]:
    """Collect the indices of the atoms which are positioned on the rotation axis.

    Args:
        dof_atoms: Atoms object that containes the dof atoms.
        axis_position: Position vector to a point in space which is part of
            the rotation axis.
        axis_vector_axis: Axis which defines the direction of the rotation axis.
        threshold: Small value to be used as upper limit condition to mark the
            rotation axis atoms.
    """

    # Get atom positions and transform them from scaled to cartesian coordiantes
    atom_positions = dof_atoms.get_positions()
    cartesian_axis_position = axis_position @ dof_atoms.cell

    # Calculate the norm dependent on the given rotation axis
    if axis_vector_axis == 0:
        y_diff = atom_positions[:, 1] - cartesian_axis_position[1]
        z_diff = atom_positions[:, 2] - cartesian_axis_position[2]
        norm = np.sqrt(y_diff**2 + z_diff**2)
    elif axis_vector_axis == 1:
        x_diff = atom_positions[:, 0] - cartesian_axis_position[0]
        z_diff = atom_positions[:, 2] - cartesian_axis_position[2]
        norm = np.sqrt(x_diff**2 + z_diff**2)
    elif axis_vector_axis == 2:
        x_diff = atom_positions[:, 0] - cartesian_axis_position[0]
        y_diff = atom_positions[:, 1] - cartesian_axis_position[1]
        norm = np.sqrt(x_diff**2 + y_diff**2)
    else:
        raise ValueError(
            "Currently only axis_vector_axis 0, 1, 2 implemented!"
        )

    # Collect the atoms close enough to the rotation axis
    axis_indices = np.where(norm < threshold)[0].tolist()
    axis_indices = None if not axis_indices else axis_indices

    return axis_indices


def get_atom_indices_between_planes(
    atoms_object: ase.atoms.Atoms,
    first_plane_position: npt.ArrayLike,
    first_plane_normal_vector: npt.ArrayLike,
    second_plane_position: npt.ArrayLike,
    second_plane_normal_vector: npt.ArrayLike,
    plane_angle: float,
) -> List[int]:
    """Collect the indices of the atoms which are positioned between two given
    planes.

    Args:
        atoms_object: Atoms object that containes the atoms.
        first_plane_position: Position vector to a point in space which is part
            of the first plane.
        first_plane_normal_vector: Vector which is normal to the first plane.
        second_plane_position: Position vector to a point in space which is
            part of the second plane.
        second_plane_normal_vector: Vector which is normal to the second plane.
        plane_angle: Angle of the two planes in relation to the unit cell basis
        vectors.
    """

    # Collect the indices of the atoms which are positioned between two given
    # planes by first collecting the indices for each plane seperately, then
    # by converting the lists into sets to utilize the intersection operation
    # that allows to return the desired indices
    between_plane_indices = list(
        set(
            get_atom_indices_above_plane(
                atoms_object=atoms_object,
                plane_position=first_plane_position,
                plane_normal_vector=first_plane_normal_vector,
                plane_angle=plane_angle,
            )
        ).intersection(
            get_atom_indices_above_plane(
                atoms_object=atoms_object,
                plane_position=second_plane_position,
                plane_normal_vector=second_plane_normal_vector,
                plane_angle=plane_angle,
            )
        )
    )

    return between_plane_indices


def get_atom_indices_inside_plane(
    atoms_object: ase.atoms.Atoms,
    plane_position: npt.ArrayLike,
    plane_normal_vector: npt.ArrayLike,
    plane_angle: float,
    eps: npt.ArrayLike,
) -> List[int]:
    """Collect the indices of the atoms which are positioned inside a given
    plane. This function is a wrapper for the get_atom_indices_between_planes()
    function.

    Args:
        atoms_object: Atoms object that containes the atoms.
        plane_position: Position vector to a point in space which is part
            of the plane.
        plane_normal_vector: Vector which is normal to the plane.
        plane_angle: Angle of the plane in relation to the unit cell basis
        vectors.
        eps: Small vector to shift specific plane positions if necessary.
    """

    # Call the get_atom_indices_between_planes() function with the same plane
    # position and the same plane normal vectors, only shifted by small amounts
    # and flipped directions to get the atoms close enough to the given plane.
    inside_plane_indices = get_atom_indices_between_planes(
        atoms_object=atoms_object,
        first_plane_position=plane_position + (-1) * eps,
        first_plane_normal_vector=plane_normal_vector,
        second_plane_position=plane_position + eps,
        second_plane_normal_vector=(-1) * plane_normal_vector,
        plane_angle=plane_angle,
    )

    return inside_plane_indices


def get_atom_indices_above_plane(
    atoms_object: ase.atoms.Atoms,
    plane_position: npt.ArrayLike,
    plane_normal_vector: npt.ArrayLike,
    plane_angle: float,
) -> List[int]:
    """Collect the indices of the atoms which are positioned above a given
    plane.

    Args:
        atoms_object: Atoms object that containes the atoms.
        plane_position: Position vector to a point in space which is part
            of the plane.
        plane_normal_vector: Vector which is normal to the plane.
        plane_angle: Angle of the plane in relation to the unit cell basis
        vectors.
    """

    # If the unit cell is defined by a non-orthonormal basis, the plane can be
    # rotated by the given plane angle if it is desired.
    #
    # NOTE: This specific implementation is only usable for an angle defined
    # around the z-axis.
    S = np.array(
        [
            [1, np.cos(np.radians(plane_angle)), 0],
            [0, np.sin(np.radians(plane_angle)), 0],
            [0, 0, 1],
        ]
    )

    # To collect the indices of all atoms above the given plane, the general
    # normal form of the plane is used
    return [
        atom.index
        for atom in atoms_object
        if np.dot(
            S @ (atom.scaled_position - plane_position),
            S @ unit_vector(plane_normal_vector),
        )
        > 0
    ]


def unit_vector(
    vector: npt.ArrayLike,
) -> npt.ArrayLike:
    """Construct the unit vector of the given vector.

    Args:
        vector: Vector for geometric operations.
    """

    # Normalize the vector
    return np.array(vector) / np.linalg.norm(vector)


def create_apply_symmetry_operation(
    transformation: str,
    position: npt.ArrayLike,
    axis: int,
    angle: float,
    symmetry_array: npt.ArrayLike,
    debug: bool = False,
) -> Callable:
    """Closure to create a symmetry operation function.

    Args:
        transformation: String that describes the desired symmetry operation.
        position: Either the position of the mirror plane or the rotation axis.
        axis: Either the direction of the plane normal vector or the rotation
            axis.
        angle: The rotation angle for the rotation operation.
        symmetry_array: Array that specifies which atoms are placed on
            special symmetry positions such as mirror planes or rotation axes,
            which atom coordinates are degrees of freedom and which are fixed
            during the evolution.
        debug: Flag that activates additional debug output.
    """

    def apply_symmetry_operation(
        atoms: ase.atoms.Atoms,
    ) -> ase.atoms.Atoms:
        """Apply a symmetry operation to an atoms object.

        Args:
            atoms: Atoms object that containes the atoms.
        """

        result = atoms.copy()

        # Apply either the reflection or the rotation operation to the given
        # atoms object, depending on the given transformation string
        if transformation == Operation.REFLECTION:
            result = apply_reflection_operation(
                atoms=atoms,
                result=result,
                plane_position=position,
                plane_normal_vector_axis=axis,
                symmetry_array=symmetry_array,
            )
        elif transformation == Operation.ROTATION:
            result = apply_rotation_operation(
                atoms=atoms,
                result=result,
                axis_position=position,
                axis_vector_axis=axis,
                rotation_angle=angle,
                symmetry_array=symmetry_array,
            )
        else:
            raise ValueError(
                "Currently only transformation "
                "'reflection', 'rotation' implemented!"
            )

        # If the debug flag is activated, the total number and the indices of
        # duplicate atoms resulting from the performed symmetry operation are
        # printed to the output stream. Correct output should show (0,) and []
        # for every applied symmetry operation
        if debug:
            to_delete = ase.geometry.get_duplicate_atoms(
                result,
                cutoff=1e-3,
                delete=False,
            )
            print(to_delete.shape, to_delete, sep="\n")

        return sort(result)

    return apply_symmetry_operation


def apply_reflection_operation(
    atoms: ase.Atoms,
    result: ase.Atoms,
    plane_position: npt.ArrayLike,
    plane_normal_vector_axis: int,
    symmetry_array: npt.ArrayLike,
) -> ase.atoms.Atoms:
    """Apply the reflection operation to an atoms object.

    Args:
        atoms: Atoms object that containes the atoms.
        result: Resulting atoms object after the reflection operation.
        plane_position: Position vector to a point in space which is part
            of the mirror plane.
        plane_normal_vector: Axis which is normal to the mirror plane.
        symmetry_array: Array that specifies which atoms are placed on
            special symmetry positions such as mirror planes or rotation axes,
            which atom coordinates are degrees of freedom and which are fixed
            during the evolution.
    """

    # Define the transformation matrix for the reflection operation depending on
    # on the given plane normal vector
    if plane_normal_vector_axis == 0:
        transformation_matrix = np.array(
            [
                [-1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )
    elif plane_normal_vector_axis == 1:
        transformation_matrix = np.array(
            [
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, 1],
            ]
        )
    elif plane_normal_vector_axis == 2:
        transformation_matrix = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, -1],
            ]
        )
    else:
        raise ValueError(
            "Currently only plane_normal_vector_axis 0, 1, 2 implemented!"
        )

    # Collect the indices of the reflection atoms, which represent the to be
    # reflected atoms before the operation.
    before_reflection_indices = np.where(
        symmetry_array["before_reflection_atom"][:, plane_normal_vector_axis]
    )[0]

    # Extract the reflection atoms for usage
    before_reflection_atoms = atoms[before_reflection_indices].copy()

    # Collect the indices of the reflected atoms, which represent the then
    # reflected atoms after the operation.
    after_reflection_indices = np.where(
        symmetry_array["after_reflection_atom"][:, plane_normal_vector_axis]
    )[0]

    # Apply the reflection operation to the reflection atoms
    scale_shift_and_transform(
        before_reflection_atoms,
        plane_position,
        transformation_matrix,
    )

    # Set the new positions of the reflection atoms at the indices of the
    # reflected atoms
    new_positions = result.get_positions()
    new_positions[after_reflection_indices] = (
        before_reflection_atoms.get_positions()
    )
    result.set_positions(new_positions)

    result.wrap()

    return result


def apply_rotation_operation(
    atoms: ase.Atoms,
    result: ase.Atoms,
    axis_position: npt.ArrayLike,
    axis_vector_axis: int,
    rotation_angle: float,
    symmetry_array: npt.ArrayLike,
) -> ase.atoms.Atoms:
    """Apply the rotation operation to an atoms object.

    Args:
        atoms: Atoms object that containes the atoms.
        result: Resulting atoms object after the rotation operation.
        axis_position: Position vector to a point in space which is part of
            the rotation axis.
        axis_vector_axis: Axis which defines the direction of the rotation axis.
        rotation_angle: Angle of the rotation operation. 360 degrees should be
            divisible by this value without a remainder.
        symmetry_array: Array that specifies which atoms are placed on
            special symmetry positions such as mirror planes or rotation axes,
            which atom coordinates are degrees of freedom and which are fixed
            during the evolution.
    """

    # Define the rotation matrix for the rotation operation depending on
    # on the given rotation axis
    if axis_vector_axis == 0:
        rotation_matrix = lambda angle: np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle), (-1) * np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)],
            ]
        )
    elif axis_vector_axis == 1:
        rotation_matrix = lambda angle: np.array(
            [
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [(-1) * np.sin(angle), 0, np.cos(angle)],
            ]
        )
    elif axis_vector_axis == 2:
        rotation_matrix = lambda angle: np.array(
            [
                [np.cos(angle), (-1) * np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
    else:
        raise ValueError(
            "Currently only axis_vector_axis 0, 1, 2 implemented!"
        )

    # Apply the symmetry operation stepwise for each iteration of the rotation
    for angle in range(rotation_angle, 360, rotation_angle):

        # Collect the indices of the rotation atoms, which represent the to be
        # rotated atoms before the operation.
        before_rotation_indices = np.where(
            symmetry_array["before_rotation_atom"][:, axis_vector_axis]
        )[0]

        # Extract the rotation atoms for usage
        before_rotation_atoms = atoms[before_rotation_indices].copy()

        # Collect the indices of the rotated atoms, which represent the then
        # rotated atoms after the operation.
        n_rot = int(angle / rotation_angle) - 1
        after_rotation_indices = np.where(
            symmetry_array["after_rotation_atom"][:, n_rot][
                :, axis_vector_axis
            ]
        )[0]

        # Apply the rotation operation to the rotation atoms
        scale_shift_and_transform(
            before_rotation_atoms,
            axis_position,
            rotation_matrix(np.radians(angle)),
        )

        # Set the new positions of the rotation atoms at the indices of the
        # rotated atoms
        new_positions = result.get_positions()
        new_positions[after_rotation_indices] = (
            before_rotation_atoms.get_positions()
        )
        result.set_positions(new_positions)

    result.wrap()

    return result


def scale_shift_and_transform(
    atoms_object: ase.atoms.Atoms,
    center_position: npt.ArrayLike,
    transformation_matrix: npt.ArrayLike,
) -> None:
    """Scale, shift and transform a given atoms object in the context of an
    applied symmetry operation.

    Args:
        atoms_object: Atoms object that containes the atoms.
        center_position: Center position of the symmetry operation.
        transformation_matrix: Transformation matrix of the symmetry operation.
    """

    # Define translation vector from the center position
    T = center_position

    # Transform the scaled coordinates to cartesian coordinates
    P = atoms_object.cell.cartesian_positions(
        atoms_object.get_scaled_positions()
    )

    # Translate the atoms to the origin of the unit cell
    P = P + atoms_object.cell.cartesian_positions((-1) * T)

    # Perform the desired symmetry operation by the given tranformation matrix
    P = P @ transformation_matrix.T

    # Translate the atoms back to their original position in the unit cell
    P = P + atoms_object.cell.cartesian_positions(T)

    # Transform the cartesian coordinates back to scaled coordinates
    P = atoms_object.cell.scaled_positions(P)

    # Set the new positions of the atoms object
    atoms_object.set_scaled_positions(P)


def prepare_dof(
    founder_atoms: ase.atoms.Atoms,
    center_position: npt.ArrayLike,
    radius: float,
    symmetry: str,
    threshold: float,
    begin_angle: float,
    end_angle: float,
    additional_condition_vector: npt.ArrayLike,
    flip_threshold: float,
) -> Tuple[ase.atoms.Atoms, ase.atoms.Atoms, dict]:
    """Prepare the degrees of freedom for the desired symmetry operations.

    Args:
        founder_atoms: Atoms object that containes the founder structure.
        center_position: Center of the degrees of freedom sphere.
        radius: Radius of the degrees of freedom sphere.
        symmetry: String that describes the desired symmetry of the evolution.
        threshold: Additional parameter to shift specific plane positions.
        begin_angle: Angle that defines the start of the rotation atoms slice.
        end_angle: Angle that defines the end of the rotation atoms slice.
        additional_condition_vector: Additional plane normal vector for an
            optional second plane to help define the reflection atoms.
        flip_threshold: Additional parameter to flip/reflect the additional
            condition vector to the other side of the mirror plane in a
            specific angle.
    """

    # Split atoms into dof atoms and immutable atoms
    dof_atoms, immutable_atoms = split_founder_atoms(
        founder_atoms=founder_atoms,
        center_position=center_position,
        radius=radius,
    )

    # Initialize the symmetry indices array
    dtype = np.dtype(
        [
            ("dof_atom", bool),
            ("position_coord", np.float64, 3),
            ("position_fixed", bool, 3),
            ("plane_atom", bool, 3),
            ("before_reflection_atom", bool, 3),
            ("after_reflection_atom", bool, 3),
            ("axis_atom", bool, 3),
            ("before_rotation_atom", bool, 3),
            ("after_rotation_atom", bool, (5, 3)),
        ]
    )
    number_of_dof_atoms = dof_atoms.get_positions().shape[0]
    symmetry_array = np.full(number_of_dof_atoms, False, dtype=dtype)
    symmetry_array["dof_atom"] = True

    # Fill the symmetry indices array with the information which atoms
    # shoud be part of the symmetry operations and which of the atoms are
    # placed on special symmetry positions such as mirror planes or rotation
    # axes, depending on the given symmetry
    if symmetry == Symmetry.NONE:
        dof_atoms = dof_atoms.copy()
    elif symmetry == Symmetry.HORIZONTAL:
        dof_atoms, symmetry_array = prepare_dof_for_reflection(
            symmetry_array=symmetry_array,
            dof_atoms=dof_atoms,
            plane_position=center_position,
            plane_normal_vector_axis=2,  # "z"
            plane_angle=0,
            threshold=threshold,
        )
    elif symmetry == Symmetry.THREEFOLD:
        dof_atoms, symmetry_array = prepare_dof_for_rotation(
            dof_atoms=dof_atoms,
            symmetry_array=symmetry_array,
            axis_position=center_position,
            axis_vector_axis=2,  # "z"
            rotation_angle=120,
            begin_angle=begin_angle,
            end_angle=end_angle,
            threshold=threshold,
        )
    elif symmetry == Symmetry.THREEFOLD_HORIZONTAL:
        dof_atoms, symmetry_array = prepare_dof_for_rotation(
            dof_atoms=dof_atoms,
            symmetry_array=symmetry_array,
            axis_position=center_position,
            axis_vector_axis=2,  # "z"
            rotation_angle=120,
            begin_angle=begin_angle,
            end_angle=end_angle,
            threshold=threshold,
        )
        dof_atoms, symmetry_array = prepare_dof_for_reflection(
            dof_atoms=dof_atoms,
            symmetry_array=symmetry_array,
            plane_position=center_position,
            plane_normal_vector_axis=2,  # "z"
            plane_angle=0,
            threshold=threshold,
        )
    elif symmetry == Symmetry.SIXFOLD_HORIZONTAL:
        dof_atoms, symmetry_array = prepare_dof_for_rotation(
            dof_atoms=dof_atoms,
            symmetry_array=symmetry_array,
            axis_position=center_position,
            axis_vector_axis=2,  # "z"
            rotation_angle=120,
            begin_angle=begin_angle,
            end_angle=end_angle,
            threshold=threshold,
        )
        dof_atoms, symmetry_array = prepare_dof_for_reflection(
            dof_atoms=dof_atoms,
            symmetry_array=symmetry_array,
            plane_position=center_position,
            plane_normal_vector_axis=0,  # "x"
            plane_angle=120,
            threshold=threshold,
            additional_condition_vector=additional_condition_vector,
            flip_threshold=flip_threshold,
        )
        dof_atoms, symmetry_array = prepare_dof_for_reflection(
            dof_atoms=dof_atoms,
            symmetry_array=symmetry_array,
            plane_position=center_position,
            plane_normal_vector_axis=2,  # "z"
            plane_angle=0,
            threshold=threshold,
        )
    else:
        raise ValueError(
            "Currently only symmetry "
            "'none', 'horizontal', 'threefold', "
            "'threefold_horizontal', 'sixfold_horizontal' implemented!"
        )

    return dof_atoms, immutable_atoms, symmetry_array


def create_pipeline(
    dof_atoms: ase.atoms.Atoms,
    immutable_atoms: ase.atoms.Atoms,
    symmetry: str,
    symmetry_array: npt.ArrayLike,
    center_position: npt.ArrayLike,
) -> DofPipelineSym:
    """Prepare the pipeline for the desired symmetry operations.

    Args:
        dof_atoms: Atoms object that containes the dof atoms.
        immutable_atoms: Atoms object that containes the immutable atoms.
        symmetry: String that describes the desired symmetry of the evolution.
        symmetry_array: Array that specifies which atoms are placed on
            special symmetry positions such as mirror planes or rotation axes,
            which atom coordinates are degrees of freedom and which are fixed
            during the evolution.
        center_position: Center position of the symmetry operations.
    """

    # Translate the degrees of freedom into an atoms object
    dof_to_atoms = DofToAtomsSym(
        template_atoms=dof_atoms,
        symmetry_array=symmetry_array,
    )

    # Construct the pipeline object with all desired symmetry operations,
    # depending on the given symmetry
    if symmetry == Symmetry.NONE:
        embed_in_anchor = AnchorToImmutable(
            immutable_atoms=immutable_atoms,
            sort_atoms=True,
        )
        pipeline = DofPipelineSym(
            anchor_to_immutable=embed_in_anchor,
            dof_to_atoms=dof_to_atoms,
        )
    elif symmetry == Symmetry.HORIZONTAL:
        apply_reflection_operation = create_apply_symmetry_operation(
            transformation=Operation.REFLECTION,
            position=center_position,
            axis=2,  # "z"
            angle=None,
            symmetry_array=symmetry_array,
        )
        embed_in_anchor = AnchorToImmutable(
            immutable_atoms=immutable_atoms,
            sort_atoms=True,
        )
        pipeline = DofPipelineSym(
            anchor_to_immutable=embed_in_anchor,
            apply_first_symmetry_operation=apply_reflection_operation,
            dof_to_atoms=dof_to_atoms,
        )
    elif symmetry == Symmetry.THREEFOLD:
        apply_rotation_operation = create_apply_symmetry_operation(
            transformation=Operation.ROTATION,
            position=center_position,
            axis=2,  # "z"
            angle=120,
            symmetry_array=symmetry_array,
        )
        embed_in_anchor = AnchorToImmutable(
            immutable_atoms=immutable_atoms,
            sort_atoms=True,
        )
        pipeline = DofPipelineSym(
            anchor_to_immutable=embed_in_anchor,
            apply_first_symmetry_operation=apply_rotation_operation,
            dof_to_atoms=dof_to_atoms,
        )
    elif symmetry == Symmetry.THREEFOLD_HORIZONTAL:
        apply_z_reflection_operation = create_apply_symmetry_operation(
            transformation=Operation.REFLECTION,
            position=center_position,
            axis=2,  # "z"
            angle=None,
            symmetry_array=symmetry_array,
        )
        apply_z_rotation_operation = create_apply_symmetry_operation(
            transformation=Operation.ROTATION,
            position=center_position,
            axis=2,  # "z"
            angle=120,
            symmetry_array=symmetry_array,
        )
        embed_in_anchor = AnchorToImmutable(
            immutable_atoms=immutable_atoms,
            sort_atoms=True,
        )
        pipeline = DofPipelineSym(
            anchor_to_immutable=embed_in_anchor,
            apply_second_symmetry_operation=apply_z_rotation_operation,
            apply_first_symmetry_operation=apply_z_reflection_operation,
            dof_to_atoms=dof_to_atoms,
        )
    elif symmetry == Symmetry.SIXFOLD_HORIZONTAL:
        apply_z_reflection_operation = create_apply_symmetry_operation(
            transformation=Operation.REFLECTION,
            position=center_position,
            axis=2,  # "z"
            angle=None,
            symmetry_array=symmetry_array,
        )
        apply_x_reflection_operation = create_apply_symmetry_operation(
            transformation=Operation.REFLECTION,
            position=center_position,
            axis=0,  # "x"
            angle=None,
            symmetry_array=symmetry_array,
        )
        apply_z_rotation_operation = create_apply_symmetry_operation(
            transformation=Operation.ROTATION,
            position=center_position,
            axis=2,  # "z"
            angle=120,
            symmetry_array=symmetry_array,
        )
        embed_in_anchor = AnchorToImmutable(
            immutable_atoms=immutable_atoms,
            sort_atoms=True,
        )
        pipeline = DofPipelineSym(
            anchor_to_immutable=embed_in_anchor,
            apply_third_symmetry_operation=apply_z_rotation_operation,
            apply_second_symmetry_operation=apply_x_reflection_operation,
            apply_first_symmetry_operation=apply_z_reflection_operation,
            dof_to_atoms=dof_to_atoms,
        )
    else:
        raise ValueError(
            "Currently only symmetry "
            "'none', 'horizontal', 'threefold', "
            "'threefold_horizontal', 'sixfold_horizontal' implemented!"
        )

    return pipeline


def prepare_dof_and_pipeline_sym(
    founder_file: pathlib.Path,
    scaled_center: Tuple[float, float, float] = None,
    radius: float = None,
    symmetry_args: dict = None,
) -> Tuple[npt.ArrayLike, Callable, ase.atoms.Atoms, dict, dict]:
    """Get DOF vector and pipeline for reconstruction.

    Define and apply symmetry operations to DOF atoms.

    Args:
        founder_file: Full path to founder POSCAR.
        scaled_center: Center point of DOF sphere, in scaled coordinates.
        radius: Radius of DOF sphere.
        symmetry_args: Dictionary containing all arguments for the symmetry
            operations.
    """

    # Read the founder file
    founder_atoms = ase.io.read(founder_file)

    # Prepare dof
    dof_atoms, immutable_atoms, symmetry_array = prepare_dof(
        founder_atoms=founder_atoms,
        center_position=scaled_center,
        radius=radius,
        symmetry=symmetry_args["symmetry"],
        threshold=symmetry_args["threshold"],
        begin_angle=symmetry_args["begin_angle"],
        end_angle=symmetry_args["end_angle"],
        additional_condition_vector=symmetry_args[
            "additional_condition_vector"
        ],
        flip_threshold=symmetry_args["flip_threshold"],
    )

    # Get degrees of freedom from atoms
    dof, symmetry_array = create_dof_from_atoms(
        dof_atoms=dof_atoms,
        symmetry_array=symmetry_array,
    )

    # Create pipeline
    pipeline = create_pipeline(
        dof_atoms=dof_atoms,
        immutable_atoms=immutable_atoms,
        symmetry=symmetry_args["symmetry"],
        symmetry_array=symmetry_array,
        center_position=scaled_center,
    )

    # Transform dof
    transform_dof = create_transform_dof(pipeline)

    return (
        dof,
        transform_dof,
        dof_atoms,
        symmetry_array,
    )


def get_view_indices_sym(
    symmetry_array: npt.ArrayLike,
) -> List[int]:
    """Collect indices of dof atoms for debugging and visualization purposes.

    Args:
        symmetry_array: Array that specifies which atoms are placed on
            special symmetry positions such as mirror planes or rotation axes,
            which atom coordinates are degrees of freedom and which are fixed
            during the evolution.
    """

    # Collect indices of dof atoms by checking the symmetry indices
    view_indices = np.where(symmetry_array["dof_atom"])[0]

    return view_indices
