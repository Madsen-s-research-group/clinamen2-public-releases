"""Sample evolution of an LJ cluster.

    References:
        [1] The Cambridge Cluster Database, D. J. Wales, J. P. K. Doye,
        A. Dullweber, M. P. Hodges, F. Y. Naumkin F. Calvo, J. Hernández-Rojas
        and T. F. Middleton, URL http://www-wales.ch.cam.ac.uk/CCD.html.
        [2]: L. Martínez, R. Andrade, E. G. Birgin, J. M. Martínez.
        J. Comput. Chem., 30(13):2157-2164, 2009.

"""
import pathlib
from typing import Tuple

import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm

from clinamen2.cmaes.cmaes_criteria import (
    EqualFunValuesCriterion,
    TolXUpCriterion,
)
from clinamen2.cmaes.params_and_state import (
    AlgorithmParameters,
    AlgorithmState,
    create_resample_and_evaluate,
    create_sample_from_state,
    create_update_algorithm_state,
)
from clinamen2.cmaes.termination_criterion import (
    CriteriaOr,
    StaleLossCriterion,
    StaleStepCriterion,
)
from clinamen2.utils.file_handling import CMAFileHandler
from clinamen2.utils.lennard_jones import (
    PositionException,
    create_evaluate_lj_potential,
    create_position_filter,
    get_max_span_from_lj_cluster,
    lj_argparse,
)
from clinamen2.utils.script_functions import cma_setup
from clinamen2.utils.structure_setup import (
    FilterEvalWorkflow,
    create_filter_eval_workflow,
    place_atoms_packmol,
    place_atoms_random_cube,
    place_atoms_random_sphere,
)

WALES_PATH = pathlib.Path(pathlib.Path.home() / "Wales")


def lj_evolution(
    step_size: float = 1.0,
    pop_size=None,
    generations: int = 1000,
    label: str = "trial",
    random_seed: int = 0,
    wales_path=pathlib.Path(pathlib.Path.home() / "Wales"),
    lj_atoms: int = 13,
    lj_identifier=None,
    lj_init_config="cube",
    packmol_executable: str = None,
    packmol_tolerance: float = 1.0,
    packmol_side_length: float = 5.0,
    packmol_seed: int = -1,
    position_bounds: npt.ArrayLike = None,
    initial_mean: npt.ArrayLike = None,
    save_nth=1,
    quiet=False,
    continue_evolution: bool = False,
    continue_with_parameters: AlgorithmParameters = None,
    continue_from_checkpoint: Tuple[
        AlgorithmState, npt.ArrayLike, npt.ArrayLike, dict
    ] = None,
) -> Tuple[str, float, AlgorithmParameters, AlgorithmState, bool]:
    """Run an evolution of an LJ cluster.

    Args:
        step_size: Initital step size for the CMA-ES. Default is 1.0.
        pop_size: Initial population size for the CMA-ES. Default is None.
        generation: Maximum number of generations to run.
        label: Textlabel of the evolution. Name of results sub directory
            to be created in 'examples/'. Default is "trial". Existing data
            will be overwritten but not deleted beforehand.
        random_seed: Random seed used for initial configuration (cube, sphere)
            and CMA-ES. Default is 0.
        wales_path: Path to the Wales potential data. Default is '~/Wales'.
        lj_atoms: Number of atoms in the cluster. Default is 13.
        lj_identifier: Additional identifier of a specific configuration. [1]
            For example "i" for "38i". Default is None.
        lj_init_config: Initial configuration type to start from. The options
            are 'cube' (randomly placed within a cube), 'sphere' (randomly
            placed within a sphere) and 'packmol' (randomly places by packmol).
            Default is 'cube'.
        packmol_executable: Packmol executable with full path. Default is None.
        packmol_tolerance: Packmol parameter. [1]
        packmol_side_length: Packmol parameter. [1]
        packmol_seed: Packmol parameter. [1]
        position_bounds: Bounds for atom positions.
        initial_mean: Alternative founder for the evolution.
        save_nth: Every nth generation is saved to json.
        quiet: If True, only the last generation is saved to json.
            Default is False.
        continue_evolution: If set to yes, a continuation is attempted.
        continue_with_parameters: Parameters to continue with.
        continue_from_checkpoint: Generation data to continue with, includes
            the state, population and loss.

    Returns:
        tuple
            - Filename + path of last mean
            - Minimum loss of last generation
            - AlgorithmParameters
            - AlgorithmState of last generation
            - Flag indicating if the global solution has been found
    """
    lj_cluster, eval_lj, eval_lj_batch = create_evaluate_lj_potential(
        n_atoms=lj_atoms,
        identifier=lj_identifier,
        wales_path=wales_path,
    )
    ref_val = float(eval_lj(lj_cluster)[0])
    print(f"Ref val is {ref_val}")
    max_span = get_max_span_from_lj_cluster(
        lj_cluster=lj_cluster, verbose=False
    )
    print(f"Random seed is {random_seed}.")
    if continue_evolution:
        parameters = continue_with_parameters
        state, population, loss, _, _ = continue_from_checkpoint
    else:
        if lj_init_config == "cube":
            side_length = max_span * 2.0
            print(f"Side length for random cube computed as {side_length}.")
            lj_cluster = place_atoms_random_cube(
                n_atoms=lj_atoms,
                side_length=side_length,
                random_seed=random_seed,
            )
        elif lj_init_config == "sphere":
            radius = max_span
            print(f"Radius for random sphere computed as {radius}.")
            lj_cluster = place_atoms_random_sphere(
                n_atoms=lj_atoms, radius=radius, random_seed=random_seed
            )
        elif lj_init_config == "packmol":
            print("Generate configuration using packmol.")
            lj_cluster = place_atoms_packmol(
                n_atoms=lj_atoms,
                side_length=packmol_side_length,
                tolerance=packmol_tolerance,
                exec_string=packmol_executable,
                random_seed=packmol_seed,
            )
        else:
            raise NotImplementedError(
                f"Init config {lj_init_config} not implemented."
            )
        flat_lj_cluster = (
            initial_mean if initial_mean is not None else lj_cluster.flatten()
        )
        founder = np.asarray(flat_lj_cluster)

        parameters, state = cma_setup(
            mean=founder,
            step_size=step_size,
            run_seed=random_seed,
            pop_size=pop_size,
        )
    update_state = create_update_algorithm_state(parameters)
    sample_individuals = create_sample_from_state(parameters)
    bound_factor = 2.0
    if position_bounds is None:
        bound = bound_factor * max_span
        position_bounds = np.asarray(
            [
                [-bound, bound],
                [-bound, bound],
                [-bound, bound],
            ]
        )
    else:
        position_bounds = np.reshape(
            a=np.asarray(position_bounds), newshape=(3, 2)
        )
    single_filter, batch_filter = create_position_filter(
        position_bounds=position_bounds,
        exception=PositionException,
    )
    eval_workflow = FilterEvalWorkflow(
        filter=single_filter, evaluate_loss=eval_lj
    )
    eval_workflow = create_filter_eval_workflow(eval_workflow)
    eval_batch_pipeline = FilterEvalWorkflow(
        filter=batch_filter, evaluate_loss=eval_lj_batch
    )
    eval_batch_pipeline = create_filter_eval_workflow(eval_batch_pipeline)

    sample_and_evaluate = create_resample_and_evaluate(
        sample_individuals=sample_individuals,
        evaluate_batch=eval_batch_pipeline,
        evaluate_single=eval_workflow,
    )

    termination_criteria = [
        EqualFunValuesCriterion(parameters=parameters, atol=1e-10),
        TolXUpCriterion(parameters=parameters, interpolative=False),
        StaleLossCriterion(
            parameters=parameters, threshold=1e-10, generations=100
        ),
        StaleStepCriterion(
            parameters=parameters, threshold=1e-10, generations=100
        ),
    ]
    termination_criterion = CriteriaOr(
        parameters=parameters, criteria=termination_criteria
    )
    if continue_evolution:
        termination_criterion_state = termination_criterion.init()
        termination_criterion_state = termination_criterion.update(
            criterion_state=termination_criterion_state,
            state=state,
            population=population,
            loss=loss,
        )
        last_gen = continue_from_checkpoint[0].generation
    else:
        termination_criterion_state = termination_criterion.init()
        last_gen = 0

    # save evolution
    run_label = label
    target_dir = pathlib.Path.cwd() / run_label
    target_dir.mkdir(parents=True, exist_ok=True)
    handler = CMAFileHandler(target_dir=target_dir, label=run_label)
    handler.save_evolution(
        initial_parameters=parameters,
        initial_state=state,
    )

    ref_found = False
    with tqdm(
        range(generations),
        bar_format="{l_bar}{bar}{r_bar}",
        postfix={"loss": 0.0, "step": 0.0, "std": 0.0},
    ) as t:
        for g in range(generations):
            generation, state, loss, _, _ = sample_and_evaluate(
                state,
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
            if (
                (not (g % save_nth) and not quiet)
                or (g == generations)
                or terminate
                or np.isclose(ref_val, loss.min(), atol=1e-5)
            ):
                handler.save_generation(
                    current_state=state,
                    population=generation,
                    loss=loss,
                    termination_state=termination_criterion_state,
                )
            t.set_postfix(
                {
                    "loss": loss.min(),
                    "step": state.step_size,
                    "std": loss.std(),
                }
            )
            t.update(1)
            if np.isclose(ref_val, loss.min(), atol=1e-5):
                ref_found = True
                terminate = True
                print(f"Reference value {ref_val} found (atol={1e-5}).")
            if terminate:
                print(f"Termination criterion met after {g} generations.")
                break

    # write index of last generation to evolution
    handler.update_evolution(additional=dict(last_gen=last_gen))

    print(
        f"Loss {loss.min()} for individual {loss.argmin()} in generation {g}."
    )

    with open(str(target_dir / run_label) + "_result", "w") as res_file:
        res_file.write(f"Loss {loss.min()} in generation {g}.")

    save_mean = str(target_dir / run_label) + "_mean"
    np.savetxt(save_mean, state.mean)
    print("Last mean written to file.")
    return save_mean, loss.min(), parameters, state, ref_found


if __name__ == "__main__":
    args = lj_argparse()
    print(f"argparse arguments: {args}")

    print("\n----------------------------")
    mean = None

    wales_path = (
        WALES_PATH
        if args.wales_path is None
        else pathlib.Path(args.wales_path)
    )

    if args.continue_evolution:
        try:
            handler = CMAFileHandler(
                target_dir=pathlib.Path.cwd() / args.label, label=args.label
            )
            (
                loaded_parameters,
                _,
                _,
                information,
            ) = handler.load_evolution()
        except:
            print(
                f"No data found for label {args.label} in "
                f"{pathlib.Path.cwd() / args.label}."
            )
            raise
        generation = args.generation_checkpoint
        if generation is None:
            try:
                generation = information["last_gen"]
            except KeyError:
                print(
                    "No field 'last_gen' in evolution data. "
                    "Please provide input '--generation_checkpoint'."
                )
                raise
        try:
            loaded_generation = handler.load_generation(generation=generation)
        except:
            print(
                f"No data found for generation {generation} in "
                f"{pathlib.Path.cwd() / args.label}."
            )
            raise
        print(
            f"Continue evolution '{args.label}' from generation {generation}."
        )
    else:
        loaded_parameters = None
        loaded_generation = None

    lj_evolution(
        step_size=args.step_size,
        generations=args.generations,
        label=args.label,
        random_seed=args.random_seed,
        lj_atoms=args.atom_count,
        lj_identifier=args.identifier,
        lj_init_config=args.configuration,
        pop_size=args.pop_size,
        position_bounds=args.bounds,
        initial_mean=mean,
        quiet=args.quiet,
        wales_path=wales_path,
        packmol_executable=args.packmol_executable,
        packmol_tolerance=args.packmol_tolerance,
        packmol_side_length=args.packmol_side_length,
        packmol_seed=args.packmol_seed,
        continue_evolution=args.continue_evolution,
        continue_with_parameters=loaded_parameters,
        continue_from_checkpoint=loaded_generation,
    )
