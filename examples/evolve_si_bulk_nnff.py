"""Evolution example: Si bulk with NNFF"""
import pathlib
import sys
from typing import Tuple

import jax
import numpy as np
from neuralil.committees.model import Committee
from neuralil.model import NeuralIL, ResNetCore
from packaging import version
from tqdm import tqdm

from clinamen2.cmaes.cmaes_criteria import (
    EqualFunValuesCriterion,
    TolXUpCriterion,
)
from clinamen2.cmaes.params_and_state import (
    create_sample_and_evaluate,
    create_sample_and_sequential_evaluate,
    create_sample_from_state,
    create_update_algorithm_state,
)
from clinamen2.cmaes.termination_criterion import (
    CriteriaOr,
    StaleLossCriterion,
    StaleStepCriterion,
)
from clinamen2.utils.file_handling import CMAFileHandler
from clinamen2.utils.jax_data import JSONEncoderwithJNP
from clinamen2.utils.neuralil_evaluation import (
    NeuralILInputPipeline,
    create_batch_input_pipeline,
    create_input_pipeline,
    create_neuralil_calc_energy,
    get_ensemble_model,
    load_neuralil_model,
)
from clinamen2.utils.script_functions import (
    cma_parser,
    cma_setup,
    generate_result_figures,
)
from clinamen2.utils.structure_setup import (
    bias_covariance_matrix_r as bias_covariance_matrix,
)
from clinamen2.utils.structure_setup import (
    create_split_atom,
    prepare_dof_and_pipeline,
)

SORTED_ELEMENTS = sorted(["Si"])
STANDARD_MODE = "standard"
N_ENSEMBLE = 5
FOUNDER_FILE = pathlib.Path.cwd() / "data" / "si" / "POSCAR_bulk"
REQUIRED_JAX_VERSION = "0.4.10"

if version.parse(jax.__version__) < version.parse(REQUIRED_JAX_VERSION):
    sys.exit(
        f"Dependency issue: "
        f"JAX version needs to be at least {REQUIRED_JAX_VERSION}, "
        f"detected {jax.__version__}.\n"
    )


def run_evolution(
    founder_file: pathlib.Path,
    model: str,
    run_label: str = "",
    run_seed: int = 0,
    generations: int = 50,
    initial_step_size: float = 0.5,
    batch: bool = False,
    save_nth: int = 1,
    scaled_center: Tuple[float, float, float] = None,
    radius: float = None,
    c_r: float = None,
    sigma_cov: float = None,
) -> int:
    """Perform evolution with given parameters.

    Args:
        - founder_file: Full path to POSCAR of founder structure.
        - model: NNFF model file (in ./trained_models).
        - run_label: Name of the evolution run. Also used to name the
            subfolder containing the results.
        - run_seed: Random number seed for initialization. Default is 0.
        - generations: Maximum number of generations to run. Default is 50.
        - initial_step_size: CMA-ES parameter step size. Default is 0.5.
        - batch: Boolean flag to switch between vmapped öpss evaluation (True)
            and sequential evaluation (False).
        - save_nth: Every n-th generation will be saved as a json file. Default
            is 1.
        - scaled_center: Point to center the sphere limiting the degrees of
            freedom on, values must be in [0, 1]. Default is None.
        - radius: Radius of sphere limiting the degrees of freedom. Default is
            None.
        - c_r: Prefactor for covariance matrix biasing.
        - sigma_cov: sigma for Gauss covariance matrix biasing.

    Returns:
        Index of last generation. Generation indices start at 1, with 0
            representing the founder.
    """

    print("\n----------------------------")
    print(
        f"Perform {run_label} for {generations} generations "
        f"starting from founder {founder_file}."
    )
    print("\n----------------------------")

    dof, transform_dof, dof_atoms = prepare_dof_and_pipeline(
        founder_file, scaled_center=scaled_center, radius=radius
    )

    split_atom = create_split_atom(sorted_elements=SORTED_ELEMENTS)
    pipeline = NeuralILInputPipeline(
        clinamen_pipeline=transform_dof, neuralil_pipeline=split_atom
    )
    if batch:
        input_pipeline = create_batch_input_pipeline(pipeline=pipeline)
    else:
        input_pipeline = create_input_pipeline(pipeline=pipeline)

    if scaled_center is not None and c_r is not None:
        initial_cholesky_factor = bias_covariance_matrix(
            atoms=dof_atoms,
            scaled_position=scaled_center,
            c_r=c_r,
            dimension=dof.shape[0],
        )
    else:
        initial_cholesky_factor = None

    parameters, state = cma_setup(
        mean=dof,
        step_size=initial_step_size,
        run_seed=run_seed,
        initial_cholesky_factor=initial_cholesky_factor,
    )
    update_state = create_update_algorithm_state(parameters)
    sample_individuals = create_sample_from_state(parameters)

    print(
        f"Starting evolution with dimension {dof.shape[0]} "
        f"and population size {parameters.pop_size}"
    )

    model_path = pathlib.Path(pathlib.Path.cwd() / "trained_models" / model)
    model, model_info = load_neuralil_model(
        pickled_model=model_path,
        max_neighbors=15,
        core_class=ResNetCore,
        model_class=NeuralIL,
    )
    ensemble = get_ensemble_model(
        individual_model=model,
        n_ensemble=N_ENSEMBLE,
        ensemble_class=Committee,
    )
    evaluate_loss, evaluate_batch_loss = create_neuralil_calc_energy(
        dynamics_model=ensemble, model_info=model_info
    )
    if batch:
        sample_and_evaluate = create_sample_and_evaluate(
            sample_individuals=sample_individuals,
            evaluate_loss=evaluate_batch_loss,
            input_pipeline=input_pipeline,
        )
    else:
        sample_and_evaluate = create_sample_and_sequential_evaluate(
            sample_individuals=sample_individuals,
            evaluate_loss=evaluate_loss,
            input_pipeline=input_pipeline,
        )

    # save evolution
    target_dir = pathlib.Path.cwd() / run_label
    target_dir.mkdir(parents=True, exist_ok=True)
    handler = CMAFileHandler(
        target_dir=pathlib.Path.cwd() / run_label, label=run_label
    )

    termination_criteria = [
        EqualFunValuesCriterion(parameters=parameters, atol=1e-5),
        TolXUpCriterion(parameters=parameters, interpolative=False),
        StaleLossCriterion(
            parameters=parameters, threshold=1e-4, generations=100
        ),
        StaleStepCriterion(
            parameters=parameters, threshold=1e-4, generations=100
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
            "scaled_center": scaled_center,
            "radius": radius,
            "c_r": c_r,
            "sigma_cov": sigma_cov,
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
            generation, state, loss = sample_and_evaluate(state=state)

            idx = np.argsort(loss)
            state = update_state(state, generation[idx])
            # save every nth generation and the last one
            if not (g % save_nth) or (g == generations):
                handler.save_generation(
                    current_state=state,
                    population=generation,
                    loss=loss,
                    termination_state=termination_criterion_state,
                    json_encoder=JSONEncoderwithJNP,
                )
            t.set_postfix(
                {
                    "loss": loss.min(),
                    "step": state.step_size,
                    "std": loss.std(),
                }
            )
            t.update(1)
            last_gen = g

    print(
        f"Minimum loss {loss.min()} for individual "
        f"after {last_gen} generations."
    )

    # write index of last generation to evolution
    handler.update_evolution(additional=dict(last_gen=last_gen))

    return last_gen


if __name__ == "__main__":
    parser = cma_parser()
    parser.add_argument(
        "--batch",
        action="store_true",
        help="If flag is present, NN eval is vmapped.",
    )
    parser.add_argument(
        "--plot_mean",
        action="store_true",
        help="If flag is present, figures are created from output.",
    )
    parser.add_argument(
        "--scaled_center",
        type=float,
        nargs="+",
        help="Scaled center as x y z",
        default=None,
    )
    parser.add_argument(
        "--radius",
        type=float,
        help="Radius of sphere around scaled_center.",
        default=None,
    )
    parser.add_argument(
        "--c_r",
        type=float,
        help="c_r parameter for covariance matrix bias.",
        default=20.0,
    )
    parser.add_argument(
        "--sigma_cov",
        type=float,
        help="sigma parameter for covariance matrix bias.",
        default=1.25,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="NNFF file to use for loss evaluation (in ./trained_models)",
        default="model_si_committee_51.pkl",
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
    evolution_args = {
        "founder_file": FOUNDER_FILE,
        "model": args.model,
        "run_label": args.label,
        "generations": args.generations,
        "run_seed": args.random_seed,
        "initial_step_size": args.step_size,
        "save_nth": args.save_nth,
        "batch": args.batch,
        "scaled_center": args.scaled_center,
        "radius": args.radius,
        "c_r": args.c_r,
        "sigma_cov": args.sigma_cov,
    }
    last_gen = run_evolution(**evolution_args)

    if args.plot_mean:
        try:
            generate_result_figures(
                label=args.label,
                input_dir=pathlib.Path.cwd() / args.label,
                generation_bounds=(0, last_gen),
                output_dir=pathlib.Path.cwd() / args.label,
            )
        except:
            print("Figure creation failed at least partly.")
            raise
