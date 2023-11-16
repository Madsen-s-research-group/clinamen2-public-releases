"""Perform an optimization trial on test functions."""
import argparse
import json
import pathlib
from typing import Callable, Tuple, Literal

import ase
import numpy as np
import numpy.random
import numpy.typing as npt
from ase.io import read
from tqdm import tqdm

from clinamen2.cmaes.cmaes_criteria import (
    EqualFunValuesCriterion,
    TolXUpCriterion,
)
from clinamen2.cmaes.params_and_state import (
    create_sample_and_run,
    create_sample_from_state,
    create_update_algorithm_state,
)
from clinamen2.cmaes.termination_criterion import CriteriaOr
from clinamen2.runner.basic_runner import ScriptRunner
from clinamen2.utils.file_handling import CMAFileHandler
from clinamen2.utils.script_functions import (
    cma_parser,
    cma_setup,
    generate_result_figures,
)
from clinamen2.utils.structure_setup import (
    DofToAtoms,
    place_atoms_random_cube,
    place_atoms_random_sphere,
)


def prepare_dof_and_pipeline(
    founder_filename: str,
    randomize_positions: bool = False,
    random_positions_limit: float = 0.5,
    random_seed: int = 0,
) -> Tuple[npt.ArrayLike, Callable]:
    """Cluster from POSCAR

    Args:
        founder_filename: Full path to POSCAR containing founder.
        randomize_positions: If set to True, the atom positions read from the
            founder are replaced by randomly drawn positions. Default is False.
        random_positions_limit: Defines within how much of the inner region of
            the original cell the new positions are drawn. Default is 0.5,
            i.e., the inner region limited by half the original side length is
            used. Current iteration only works for a cubic cell.
        random_seed: Seed for random number generator used to draw positions.

    Returns:
        Tuple containing
        - Degrees of freedom (CMA-ES input).
        - Function to recreate atoms object from dof.
    """
    atoms = read(founder_filename)
    dof = atoms.get_positions().flatten()

    if randomize_positions is not None:
        if random_positions_limit > 1.0:
            raise ValueError(
                "random_positions_limit may not be " "greater than 1.0"
            )
        side_length = atoms.cell[0][0]
        # function draws positions around [0, 0, 0]
        if randomize_positions == "cube":
            dof = (
                place_atoms_random_cube(
                    n_atoms=atoms.positions.shape[0],
                    side_length=side_length * random_positions_limit,
                    random_seed=random_seed,
                )
                + side_length * 0.5
            )  # shift to be centered in original cell
        elif randomize_positions == "sphere":
            # additional limit of 0.5, because it is the radius
            # function draws positions around [0, 0, 0]
            dof = (
                place_atoms_random_sphere(
                    n_atoms=atoms.positions.shape[0],
                    radius=side_length * random_positions_limit * 0.5,
                    random_seed=random_seed,
                )
                + side_length * 0.5
            )  # shift to be centered in original cell
        else:
            raise NotImplementedError(
                f"randomize_positions {randomize_positions} not implemented."
            )

    print(dof)
    print(f"{dof.shape[0]} degrees of freedom.")

    dof_to_atoms = DofToAtoms(template_atoms=atoms)

    return dof, dof_to_atoms


def evolution(
    founder_filename: str,
    dft_backend: Literal["nwchem", "vasp"],
    step_size: float,
    generations: int,
    label: str,
    save_nth: int = 1,
    pop_size: int = None,
    seed: int = 0,
    randomize_positions: str = None,
    random_positions_limit: float = 0.5,
):
    """Run an Ag cluster optimization with DFT."""

    dof, transform_dof = prepare_dof_and_pipeline(
        founder_filename=founder_filename,
        randomize_positions=randomize_positions,
        random_positions_limit=random_positions_limit,
        random_seed=seed,
    )

    parameters, state = cma_setup(
        mean=dof,
        step_size=step_size,
        pop_size=pop_size,
        run_seed=seed,
    )
    update_state = create_update_algorithm_state(parameters)
    sample_individuals = create_sample_from_state(parameters)

    if dft_backend == "nwchem":
        n_atoms = len(read(founder_filename))
        multiplicity = n_atoms % 2 + 1
        SCRIPT_CONFIG = {
            "pbc": False,
            "nwchem_params": {
                "label": "'calc/nwchem'",
                "dft": dict(
                    maxiter=100,
                    xc="xpbe96 cpbe96",
                    mult=multiplicity,
                    smear=0.001,
                ),
                "basis": "'3-21G'",
            },
        }
        runner_script_filename = "nwchem_script.py.j2"
        scheduler_filename = "scheduler_nwchem.json"
    elif dft_backend == "vasp":
        SCRIPT_CONFIG = {
            "vasp_params": {
                "nsw": 0,
                "gga": "'PE'",
                "pp": "'PBE'",
                "ispin": 2,
                "isym": 0,
                "ismear": 0,
                "sigma": 0.0001,
                "ediff": 1e-6,
                "nelm": 80,
                "kpts": (1, 1, 1),
                "lorbit": 11,
                "lcharg": False,
                "lwave": False,
                "ncore": 8,
            },
        }
        runner_script_filename = "vasp_script.py.j2"
        scheduler_filename = "scheduler_vasp.json"
    else:
        raise ValueError("Invalid choice of DFT backend.")

    with open(
        pathlib.Path.cwd() / "runner_scripts" / runner_script_filename,
        "r",
        encoding="utf-8",
    ) as f:
        SCRIPT_TEXT = f.read()

    runner = ScriptRunner(
        script_text=SCRIPT_TEXT,
        script_config=SCRIPT_CONFIG,
        script_run_command="python {SCRIPTFILE}",
        convert_input=transform_dof,
        scheduler_info_path=scheduler_filename,
    )

    sample_and_run = create_sample_and_run(
        sample_individuals,
        runner,
    )

    termination_criteria = [
        EqualFunValuesCriterion(parameters=parameters, atol=1e-10),
        TolXUpCriterion(parameters=parameters, interpolative=False),
    ]
    termination_criterion = CriteriaOr(
        parameters=parameters, criteria=termination_criteria
    )
    termination_criterion_state = termination_criterion.init()

    # save evolution
    run_label = label
    target_dir = pathlib.Path.cwd() / run_label
    target_dir.mkdir(parents=True, exist_ok=True)
    handler = CMAFileHandler(
        target_dir=pathlib.Path.cwd() / run_label, label=run_label
    )
    handler.save_evolution(
        initial_parameters=parameters,
        initial_state=state,
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
            information = []
            (
                generation_dict,
                state,
                loss_dict,
                information_dict,
            ) = sample_and_run(state, parameters.pop_size)
            for key, val in generation_dict.items():
                try:
                    loss.append(loss_dict[key])
                    generation.append(val)
                    try:
                        information.append(information_dict[key])
                    except KeyError:
                        information.append({})
                except KeyError:
                    pass
            generation = np.asarray(generation)
            loss = np.asarray(loss)
            idx = np.argsort(loss)
            state = update_state(state, generation[idx])
            termination_criterion_state = termination_criterion.update(
                criterion_state=termination_criterion_state,
                state=state,
                population=generation,
                loss=loss,
            )
            terminate = termination_criterion.met(termination_criterion_state)

            # save every nth generation and the last one
            if not (g % save_nth) or (g == generations) or terminate:
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
            if terminate:
                print(f"Termination criterion met after {g} generations.")
                break

    # write index of last generation to evolution
    handler.update_evolution(additional=dict(last_gen=last_gen))

    print(
        f"Loss {loss.min()} for individual\n"
        f"{loss.argmin()} in generation {last_gen}."
    )

    return last_gen


if __name__ == "__main__":
    parser = cma_parser()
    parser.add_argument(
        "-f",
        "--founder",
        type=str,
        required=True,
        help="Filename with relative path to founder.",
    )
    parser.add_argument(
        "--dft_backend",
        type=str,
        required=True,
        choices=["nwchem", "vasp"],
        help="Which DFT backend to use for loss evaluation.",
    )
    parser.add_argument(
        "--plot_mean",
        action="store_true",
        help="If flag is present, figures are created from output.",
    )
    parser.add_argument(
        "--randomize_positions",
        type=str,
        default=None,
        help="Randomize founder positions. 'cube' or 'sphere'. Default is None",
    )
    parser.add_argument(
        "--random_positions_limit",
        type=float,
        default=0.5,
        help="Portion of side length for random range.",
    )
    args, unknown = parser.parse_known_args()
    print(f"argparse arguments: {args}")
    print("\n----------------------------")

    last_gen = evolution(
        founder_filename=args.founder,
        dft_backend=args.dft_backend,
        step_size=args.step_size,
        generations=args.generations,
        label=args.label,
        save_nth=args.save_nth,
        pop_size=args.pop_size,
        seed=args.random_seed,
        randomize_positions=args.randomize_positions,
        random_positions_limit=args.random_positions_limit,
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
