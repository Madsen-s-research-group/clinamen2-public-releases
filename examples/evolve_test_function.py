"""Perform an optimization trial on test functions."""
import json
import pathlib
from typing import Tuple

import numpy as np
import numpy.random
import numpy.typing as npt
from tqdm import tqdm

from clinamen2.cmaes.cmaes_criteria import TolXUpCriterion
from clinamen2.cmaes.params_and_state import (
    AlgorithmParameters,
    AlgorithmState,
    create_sample_and_sequential_evaluate,
    create_sample_from_state,
    create_update_algorithm_state,
)
from clinamen2.cmaes.termination_criterion import CriteriaOr
from clinamen2.utils.file_handling import CMAFileHandler
from clinamen2.utils.script_functions import (
    cma_parser,
    cma_setup,
    generate_result_figures,
)
from clinamen2.test_functions import (
    create_ackley_function,
    create_cigar_function,
    create_diffpowers_function,
    create_discus_function,
    create_ellipsoid_function,
    create_rosenbrock_function,
    create_sphere_function,
)

LOSS_CUTOFF = 1e-10


def founder_setup(
    dimension,
    step_size,
    run_seed=0,
    pop_size=None,
    sphere=False,
    ackley=False,
    mult=1.0,
) -> Tuple[AlgorithmParameters, AlgorithmState, npt.ArrayLike]:
    """Create the founder mean and initialize the CMA-ES."""

    # random founder
    rng = np.random.default_rng(run_seed)
    if sphere:
        founder = rng.standard_normal(dimension)
    elif ackley:
        # founder = (rng.random(dimension) - 0.5) * mult
        founder = (rng.random(dimension)) * (mult - 1) + 1
    else:
        founder = rng.random(dimension) * mult

    parameters, state = cma_setup(
        mean=founder, step_size=step_size, run_seed=run_seed, pop_size=pop_size
    )

    return parameters, state, founder


def evolution(
    function: str = "sphere",
    dimension: int = 4,
    step_size: float = 1.0,
    generations: int = 100,
    label: str = "trial",
    seed: int = 0,
    save_nth: int = 1,
    output: str = None,
    mult: float = 1.0,
    pop_size: int = None,
    continue_evolution: bool = False,
    continue_with_parameters: AlgorithmParameters = None,
    continue_from_checkpoint: Tuple[
        AlgorithmState, npt.ArrayLike, npt.ArrayLike, dict
    ] = None,
):
    """Run an optimization to its exact solution"""
    if continue_evolution:
        parameters = continue_with_parameters
        state, population, loss, _, _ = continue_from_checkpoint
    else:
        sphere = True if function == "sphere" else False
        ackley = True if function == "ackley" else False
        (parameters, state, _) = founder_setup(
            dimension=dimension,
            step_size=step_size,
            run_seed=seed,
            sphere=sphere,
            ackley=ackley,
            mult=mult,
            pop_size=pop_size,
        )

    update_state = create_update_algorithm_state(parameters)
    sample_individuals = create_sample_from_state(parameters)
    if function == "sphere":
        fun = create_sphere_function()
    elif function == "discus":
        fun = create_discus_function()
    elif function == "rosenbrock":
        fun = create_rosenbrock_function()
    elif function == "cigar":
        fun = create_cigar_function()
    elif function == "ellipsoid":
        fun = create_ellipsoid_function()
    elif function == "diffpowers":
        fun = create_diffpowers_function()
    elif function == "ackley":
        fun = create_ackley_function()
    else:
        raise NotImplementedError(f"Function {function} not implemented.")
    evaluate_loss = fun

    sample_and_evaluate = create_sample_and_sequential_evaluate(
        sample_individuals=sample_individuals,
        evaluate_loss=evaluate_loss,
        # input_pipeline=lambda x: x.T,
    )

    run_label = label
    target_dir = pathlib.Path.cwd() / run_label
    target_dir.mkdir(parents=True, exist_ok=True)
    handler = CMAFileHandler(
        target_dir=pathlib.Path.cwd() / run_label, label=run_label
    )
    termination_criteria = [
        # EqualFunValuesCriterion(parameters=parameters, atol=1e-10),
        TolXUpCriterion(parameters=parameters, interpolative=False),
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
        handler.save_evolution(
            initial_parameters=parameters,
            initial_state=state,
        )

    with tqdm(
        range(generations),
        bar_format="{l_bar}{bar}{r_bar}",
        postfix={"loss": 0.0, "step": 0.0, "std": 0.0},
    ) as t:
        for g in tqdm(range(generations)):
            generation, state, loss = sample_and_evaluate(state)
            idx = np.argsort(loss)
            state = update_state(state, generation[idx])
            termination_criterion_state = termination_criterion.update(
                criterion_state=termination_criterion_state,
                state=state,
                population=generation,
                loss=loss,
            )
            terminate = termination_criterion.met(termination_criterion_state)

            loss = np.asarray(loss)

            # save every nth generation and the last one
            if (
                not (g % save_nth)
                or (g == generations)
                or terminate
                or loss.min() < LOSS_CUTOFF
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
            last_gen = g
            if terminate or loss.min() < LOSS_CUTOFF:
                print(f"Termination criterion met after {g} generations.")
                break

    print(
        f"Loss {loss.min()} for individual "
        f"{loss.argmin()} in generation {last_gen}."
    )

    if output is not None:
        result = {
            "function": function,
            "dimension": dimension,
            "label": label,
            "initial_step_size": step_size,
            "min_loss": float(loss.min()),
            "last_gen": last_gen,
            "pop_size": parameters.pop_size,
            "total_evals": last_gen * parameters.pop_size,
        }
        json_dict = json.dumps(result)
        with open(output, "a", encoding="UTF-8") as f:
            f.write(f"{json_dict}\n")

    # write index of last generation to evolution
    handler.update_evolution(additional=dict(last_gen=last_gen))

    return last_gen, loss.min(), parameters, state


if __name__ == "__main__":
    parser = cma_parser()
    parser.add_argument(
        "-d",
        "--dimension",
        type=int,
        default=4,
        help="Dimension of the founder.",
    )
    parser.add_argument(
        "-f",
        "--function",
        type=str,
        default="sphere",
        help="Test function to use: sphere, discus, "
        "cigar, rosenbrock, ellipsoid, diffpowers, ackley",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="File to write performance information to.",
    )
    parser.add_argument(
        "-m",
        "--mult",
        type=float,
        default=1.0,
        help="Increase range of random rounder values.",
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
    parser.add_argument(
        "--plot_mean",
        action="store_true",
        help="If flag is present, figures are created from output.",
    )
    args, unknown = parser.parse_known_args()
    print(f"argparse arguments: {args}")
    print("\n----------------------------")

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

    last_gen, _, _, _ = evolution(
        function=args.function,
        dimension=args.dimension,
        step_size=args.step_size,
        generations=args.generations,
        label=args.label,
        seed=args.random_seed,
        save_nth=args.save_nth,
        output=args.output,
        mult=args.mult,
        pop_size=args.pop_size,
        continue_evolution=args.continue_evolution,
        continue_with_parameters=loaded_parameters,
        continue_from_checkpoint=loaded_generation,
    )

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
