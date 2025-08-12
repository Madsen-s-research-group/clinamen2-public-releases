"""Evolution example: Monolayer MoS2 with MACE

Example usage:
    $ python evolve_mos2_mace.py --symmetry threefold_horizontal --random_seed 1 --pop_size 25 --step_size 0.75 --generations 1000
"""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import copy
import os
import pathlib
from pprint import pprint
from typing import Callable, NamedTuple, Tuple

import ase
import numpy as np
import numpy.typing as npt
from ase.io import write
from mace.calculators import MACECalculator
from tqdm import tqdm

from clinamen2.cmaes.cmaes_criteria import (
    EqualFunValuesCriterion,
    TolXUpCriterion,
)
from clinamen2.cmaes.params_and_state import (
    create_resample_and_evaluate,
    create_sample_from_state,
    create_update_algorithm_state,
)
from clinamen2.cmaes.termination_criterion import (
    CriteriaOr,
    StaleLossCriterion,
    StaleStepCriterion,
)
from clinamen2.utils.file_handling import CMAFileHandler, JSONEncoder
from clinamen2.utils.script_functions import cma_parser, cma_setup
from clinamen2.utils.structure_setup import (
    FilterEvalWorkflow,
    create_filter_eval_workflow,
)
from clinamen2.utils.symmetry_setup import (
    bias_covariance_matrix_r_sym,
    get_view_indices_sym,
    prepare_dof_and_pipeline_sym,
)


class LossThresholdException(Exception):
    """Exception to be raised for loss below threshold.

    This exception is raised instead of checking
    min distances.
    """

    pass


def create_lossthreshold_filter(
    loss_threshold: float = None,
    loss_threshold_exception: Exception = BaseException,
) -> Tuple[Callable, Callable]:
    """Create function to filter configurations.

    The filter is applied with regards to an absolute loss threshold.
    This is a lower bound that may not be exceeded.
    "Poor man's min distance exception."

    Args:
        loss_threshold: Absolute value of lower loss bound.
            Default is None, filters will be identity functions.
    """

    if loss_threshold is None:
        return lambda *x: x, lambda *x: x

    else:

        def batch_lossthreshold_filter(
            loss: float, additional: list, inputs: Tuple
        ) -> Tuple[float, list, Tuple]:
            for a in additional:
                check_val = a["loss"]
                if check_val < loss_threshold:
                    a["exception"] = loss_threshold_exception(
                        f"Loss {check_val} "
                        f"below threshold of {loss_threshold}."
                    )

            return loss, additional, inputs

        def single_lossthreshold_filter(
            loss: float, additional: list, inputs: Tuple
        ) -> Tuple[float, list, Tuple]:
            if loss < loss_threshold:
                additional["exception"] = loss_threshold_exception(
                    f"Loss {loss} " f"below threshold of {loss_threshold}."
                )

            return loss, additional, inputs

        return single_lossthreshold_filter, batch_lossthreshold_filter


class InputPipeline(NamedTuple):
    """Combined pipelines for input transformation.

    Args:
        placeholder_pipeline: Function pipeline performing placeholder specific
            data transformation.
        clinamen_pipeline: Function pipeline performing Clinamen2 specific
            data transformation.
    """

    placeholder_pipeline: Callable = None  # nothing
    clinamen_pipeline: Callable = None  # dof -> structure


def create_input_pipeline(pipeline: InputPipeline) -> Callable:
    """Create combined pipeline.

    Args:
        pipeline: Function pipeline for input transformation.
    """

    def input_pipeline(
        dof: npt.ArrayLike,
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        """Apply input pipeline."""
        transformed = copy.deepcopy(dof)
        transformed = transformed.reshape((-1,))

        for fun in pipeline[::-1]:
            transformed = fun(transformed) if fun is not None else transformed
        transformed.set_pbc([1, 1, 0])

        return transformed

    return input_pipeline


def create_batch_input_pipeline(pipeline: InputPipeline) -> Callable:
    """Create combined batch pipeline.

    Args:
        pipeline: Function pipeline for batched input transformation.
    """

    def input_pipeline(
        dof: npt.ArrayLike,
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        """Apply input pipeline."""
        input_values = copy.deepcopy(dof)
        transformed = []

        for t in input_values:
            for fun in pipeline[::-1]:
                t = fun(t) if fun is not None else t
                t.set_pbc([1, 1, 0])
            transformed.append(t)

        return transformed

    return input_pipeline


def create_mace_energy_and_forces(
    model_paths: list,
) -> Tuple[Callable, Callable]:
    """Create MACE calculator object and evaluation functions.

    Args:
        model_paths: List of paths to the members of the force field committee.
    """

    calc = MACECalculator(
        model_paths=model_paths,
        device="cuda",
        default_dtype="float32",
    )

    def mace_energy_and_forces(atoms) -> Tuple[list, dict]:
        at = atoms.copy()
        at.calc = calc

        energy = at.get_potential_energy()
        if len(model_paths) == 1:
            forces = at.calc.results["forces"]
        else:
            forces = at.calc.results["forces_comm"]

        return energy, {"loss": energy, "forces": forces}

    def mace_energy_and_forces_batch(atoms_list) -> Tuple[list, list]:
        energies = []
        additional = []

        for atoms in atoms_list:
            energy, info = mace_energy_and_forces(atoms)
            energies.append(energy)
            additional.append(info)

        return energies, additional

    return mace_energy_and_forces, mace_energy_and_forces_batch


def calc_uncertainties(
    population: np.ndarray,
    information: np.ndarray,
    transform_dof: Callable,
    r_cut: float = 4.0,
) -> Tuple[list, list]:
    """Calculate the local and global uncertainties from the predicted forces.

    Args:
        population: Array of all individuals of the current generation.
        information: Array containing the predicted forces.
        transform_dof: Composite function of the utilized transformations.
        r_cut: Cutoff radius for the local uncertainty environments.
    """

    local_uncertainties = []
    global_uncertainties = []

    for dof, info in zip(population, information):
        atoms = transform_dof(dof)
        atom_pos = atoms.get_positions()  # shape = (n_atoms, 3)
        forces = np.array(info["forces"])
        atomic_uncertainties = forces.std(axis=0)  # shape = (n_atoms, 3)

        radii = ase.geometry.get_distances(
            atom_pos, cell=atoms.get_cell(), pbc=atoms.get_pbc()
        )[1]
        mask = radii < r_cut
        unc_mean = atomic_uncertainties.mean(axis=-1)
        local_unc = (mask * unc_mean[np.newaxis, :]).sum(axis=1) / mask.sum(
            axis=1
        )
        local_uncertainties.append(local_unc.tolist())

        global_unc = atomic_uncertainties.mean()
        global_uncertainties.append(global_unc.tolist())

    return local_uncertainties, global_uncertainties


def run_evolution(
    founder: pathlib.Path,
    list_of_models: list = None,
    run_label: str = "",
    run_seed: int = 0,
    generations: int = 100,
    initial_step_size: float = 0.25,
    pop_size: int = None,
    save_nth: int = 1,
    c_r: float = None,
    scaled_center: Tuple[float, float, float] = None,
    radius: float = None,
    symmetry_args: dict = None,
) -> int:
    """Run an evolution with MACE."""

    print("\n----------------------------")
    print(
        f"Perform {run_label} for {generations} generations "
        f"starting from founder {founder}."
    )
    print("\nUsing committee:")
    for i, model in enumerate(list_of_models):
        print(f"{i+1}. {model}")
    print("\n----------------------------")

    (
        dof,
        transform_dof,
        dof_atoms,
        symmetry_array,
    ) = prepare_dof_and_pipeline_sym(
        founder_file=founder,
        scaled_center=scaled_center,
        radius=radius,
        symmetry_args=symmetry_args,
    )

    pipeline = InputPipeline(clinamen_pipeline=transform_dof)
    input_pipeline = create_input_pipeline(pipeline=pipeline)
    input_pipeline_batch = create_batch_input_pipeline(pipeline=pipeline)

    if scaled_center is not None and c_r is not None:
        initial_cholesky_factor = bias_covariance_matrix_r_sym(
            atoms=dof_atoms,
            scaled_position=scaled_center,
            c_r=c_r,
            dimension=dof.shape[0],
            symmetry_array=symmetry_array,
        )
    else:
        initial_cholesky_factor = None

    parameters, state = cma_setup(
        mean=dof,
        step_size=initial_step_size,
        run_seed=run_seed,
        pop_size=pop_size,
        initial_cholesky_factor=initial_cholesky_factor,
    )
    update_state = create_update_algorithm_state(parameters)
    sample_individuals = create_sample_from_state(parameters)

    print(
        f"Starting evolution with dimension {dof.shape[0]} "
        f"and population size {parameters.pop_size} "
        f"and symmetry '{symmetry_args['symmetry']}'."
    )

    list_of_models_paths = []
    for model in list_of_models:
        list_of_models_paths.append(pathlib.Path.cwd() / model)

    (
        evaluate_loss,
        evaluate_batch_loss,
    ) = create_mace_energy_and_forces(
        model_paths=list_of_models_paths,
    )

    single_filter, batch_filter = create_lossthreshold_filter(
        loss_threshold=-2e3,
        loss_threshold_exception=LossThresholdException,
    )
    eval_workflow_single = FilterEvalWorkflow(
        filter=single_filter, evaluate_loss=evaluate_loss
    )
    eval_workflow_single = create_filter_eval_workflow(eval_workflow_single)
    eval_workflow_batch = FilterEvalWorkflow(
        filter=batch_filter, evaluate_loss=evaluate_batch_loss
    )
    eval_workflow_batch = create_filter_eval_workflow(eval_workflow_batch)
    resample_and_evaluate = create_resample_and_evaluate(
        sample_individuals=sample_individuals,
        evaluate_batch=eval_workflow_batch,
        evaluate_single=eval_workflow_single,
        input_pipeline_batch=input_pipeline_batch,
        input_pipeline_single=input_pipeline,
    )

    target_dir = pathlib.Path.cwd() / run_label
    target_dir.mkdir(parents=True, exist_ok=True)
    handler = CMAFileHandler(
        target_dir=pathlib.Path.cwd() / run_label, label=run_label
    )

    termination_criteria = [
        EqualFunValuesCriterion(parameters=parameters, atol=1e-4),
        TolXUpCriterion(parameters=parameters, interpolative=False),
        StaleLossCriterion(
            parameters=parameters, threshold=1e-4, generations=50
        ),
        StaleStepCriterion(
            parameters=parameters, threshold=1e-4, generations=50
        ),
    ]
    termination_criterion = CriteriaOr(
        parameters=parameters, criteria=termination_criteria
    )
    termination_criterion_state = termination_criterion.init()

    handler.save_evolution(
        initial_parameters=parameters,
        initial_state=state,
        additional={
            "list_of_models": list_of_models,
            "founder": str(founder),
            "c_r": c_r,
            "scaled_center": scaled_center,
            "radius": radius,
            "symmetry_args": symmetry_args,
        },
    )

    last_gen = 0
    with tqdm(
        range(generations),
        bar_format="{l_bar}{bar}{r_bar}",
        postfix={"loss": 0.0, "step": 0.0, "std": 0.0},
    ) as t:
        for g in tqdm(range(generations)):
            generation = []
            loss = []
            (
                generation,
                state,
                loss,
                information,
                _,
            ) = resample_and_evaluate(
                state=state,
                n_samples=parameters.pop_size,
                return_failures=True,
            )
            last_gen = g
            idx = np.argsort(loss)
            state = update_state(state, generation[idx])
            termination_criterion_state = termination_criterion.update(
                criterion_state=termination_criterion_state,
                state=state,
                population=generation,
                loss=loss,
            )
            terminate = termination_criterion.met(termination_criterion_state)

            (
                local_uncertainties,
                global_uncertainties,
            ) = calc_uncertainties(
                population=generation,
                information=information,
                transform_dof=transform_dof,
            )

            if (not (g % save_nth)) or (g == generations - 1) or terminate:
                handler.save_generation(
                    current_state=state,
                    population=generation,
                    loss=loss,
                    termination_state=termination_criterion_state,
                    additional={
                        "information": information,
                        "local_uncertainties": local_uncertainties,
                        "global_uncertainties": global_uncertainties,
                    },
                    json_encoder=JSONEncoder,
                )
            t.set_postfix(
                {
                    "loss": loss.min(),
                    "step": state.step_size,
                    "std": loss.std(),
                }
            )
            t.update(1)
            if terminate:
                print(f"Termination criterion met after {g} generations.")
                break

            handler.update_evolution(additional={"last_gen": last_gen})

    print(
        f"Minimum loss {loss.min()} for individual "
        f"after {last_gen} generations."
    )

    return last_gen


def generate_slice_and_trajectory(
    run_label: str = None,
    visualize_slice_and_trajectory: bool = False,
) -> None:
    """Generate a POSCAR of the residual slice of the founder and
    an ASE trajectory file of the evolution.

    Args:
        run_label: Path to the directory of the evolution output files.
        visualize_trajectory: Flag to control if ASE GUI should automatically
            be opened right after generation.
    """

    handler = CMAFileHandler(
        target_dir=pathlib.Path.cwd() / run_label, label=run_label
    )
    evolution = handler.load_evolution()

    (
        dof,
        transform_dof,
        dof_atoms,
        symmetry_array,
    ) = prepare_dof_and_pipeline_sym(
        founder_file=evolution[-1]["founder"],
        scaled_center=evolution[-1]["scaled_center"],
        radius=evolution[-1]["radius"],
        symmetry_args=evolution[-1]["symmetry_args"],
    )

    print(f"\nGenerate residual slice of the founder structure ...")
    dof_atoms_slice = dof_atoms[get_view_indices_sym(symmetry_array)]

    slice_file_name = (
        pathlib.Path.cwd() / run_label / ("slice_" + run_label + ".vasp")
    )
    write(slice_file_name, images=dof_atoms_slice, format="vasp", direct=True)

    last_gen = evolution[-1]["last_gen"]
    print(f"Generate trajectory with {str(last_gen)} generations ...")

    trajectory = []
    for gen in range(1, last_gen + 1):
        generation = handler.load_generation(gen)
        index = np.argmin(generation[2])
        atoms = transform_dof(generation[1][index])
        trajectory.append(atoms)

    trajectory_file_name = (
        pathlib.Path.cwd() / run_label / ("trajectory_" + run_label + ".traj")
    )
    write(trajectory_file_name, trajectory)

    if visualize_slice_and_trajectory:
        print(f"Visualize residual slice and trajectory ...")
        os.system(f"ase gui {str(slice_file_name)}")
        os.system(f"ase gui {str(trajectory_file_name)}")


if __name__ == "__main__":

    parser = cma_parser()
    parser.add_argument(
        "--list_of_models",
        type=str,
        nargs="+",
        help="List of committee models.",
        default=[
            "trained_models/mos2_mace_committee/member_1.model",
            "trained_models/mos2_mace_committee/member_2.model",
            "trained_models/mos2_mace_committee/member_3.model",
        ],
    )
    parser.add_argument(
        "--founder",
        type=str,
        help="Path to founder POSCAR.",
        default="data/mos2/POSCAR_mos2_5x5x1_3v-s2",
    )
    parser.add_argument(
        "--c_r",
        type=float,
        help="c_r parameter for covariance matrix bias.",
        default=None,
    )
    parser.add_argument(
        "--scaled_center",
        type=float,
        nargs="+",
        help="Scaled center as x y z",
        default=[0.46666, 0.53333, 0.49085],
    )
    parser.add_argument(
        "--radius",
        type=float,
        help="Radius of sphere around scaled_center.",
        default=7.0,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Additional parameter to shift specific plane positions.",
        default=0.038,
    )
    parser.add_argument(
        "--begin_angle",
        type=float,
        help="Angle that defines the start of the rotation atoms slice.",
        default=99,
    )
    parser.add_argument(
        "--end_angle",
        type=float,
        help="Angle that defines the end of the rotation atoms slice.",
        default=14,
    )
    parser.add_argument(
        "--additional_condition_vector",
        type=float,
        nargs="+",
        help="Additional plane normal vector for an optional second plane to help define the reflection atoms.",
        default=[1.3, -2, 0],
    )
    parser.add_argument(
        "--flip_threshold",
        type=float,
        help="Additional parameter to flip/reflect the additional condition vector to the other side of the mirror plane in a specific angle.",
        default=-2,
    )
    parser.add_argument(
        "--symmetry",
        type=str,
        help="Classifies the symmetry operations that reduce the DOF.",
        default="threefold_horizontal",
    )
    parser.add_argument(
        "--open_visualization",
        action="store_true",
        help="Flag that decides if ASE GUI should be opened at the end or not.",
    )

    args, unknown = parser.parse_known_args()
    print(f"argparse arguments: {args}")
    print("\n----------------------------")
    print(f"Evolution {args.label} for {args.generations} generations.")
    scaled_center = (
        np.reshape(a=np.asarray(args.scaled_center), newshape=(3,))
        if args.scaled_center is not None
        else None
    )

    symmetry_args = {
        "threshold": args.threshold,
        "begin_angle": args.begin_angle,
        "end_angle": args.end_angle,
        "additional_condition_vector": args.additional_condition_vector,
        "flip_threshold": args.flip_threshold,
        "symmetry": args.symmetry,
    }
    evolution_args = {
        "founder": args.founder,
        "list_of_models": args.list_of_models,
        "run_label": args.label,
        "generations": args.generations,
        "run_seed": args.random_seed,
        "initial_step_size": args.step_size,
        "pop_size": args.pop_size,
        "save_nth": args.save_nth,
        "c_r": args.c_r,
        "scaled_center": scaled_center,
        "radius": args.radius,
        "symmetry_args": symmetry_args,
    }
    last_gen = run_evolution(**evolution_args)

    generate_slice_and_trajectory(
        run_label=args.label,
        visualize_slice_and_trajectory=args.open_visualization,
    )
