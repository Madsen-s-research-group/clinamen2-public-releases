"""Example application of the BIPOP-CMA-ES.

    N. Hansen. ACM-GECCO, 2009 (inria-00382093).
    I. Loshchilov. CEC, 2013 (hal-00823880).
"""
import argparse
import json
import pathlib
from dataclasses import asdict
from typing import Tuple

import numpy as np
import numpy.random
from evolve_lj_cluster_jax import lj_evolution

from clinamen2.cmaes.params_and_state import (
    AlgorithmParameters,
    AlgorithmState,
)
from clinamen2.utils.bipop_restart import (
    bipop_init,
    bipop_next_restart,
    bipop_update,
)
from clinamen2.utils.file_handling import JSONEncoder
from clinamen2.utils.lennard_jones import lj_argparse

LABEL = "BIPOP_LJ"

LJ_IDENTIFIER = None
WALES_PATH = pathlib.Path(pathlib.Path.home() / "Wales")


def lj_evo_call(
    step_size: float,
    label: str,
    pop_size: int,
    seed: int,
    parsed_args: argparse.Namespace,
) -> Tuple[str, float, AlgorithmParameters, AlgorithmState, bool]:
    """Wraps call to 'lj_evolution()'.

    Args:
        step_size: Initital step size for the CMA-ES. Default is 1.0.
        pop_size: Initial population size for the CMA-ES. Default is None.
        generation: Maximum number of generations to run.
        label: Textlabel of the evolution. Name of results sub directory
            to be created in 'examples/'. Default is "trial". Existing data
            will be overwritten but not deleted beforehand.
        seed: Random seed.
        parsed_args: Evolution parameters from command line.

    Returns:
        Result of lj_evolution() call.
    """

    wales_path = (
        WALES_PATH
        if parsed_args.wales_path is None
        else pathlib.Path(parsed_args.wales_path)
    )

    return lj_evolution(
        step_size=step_size,
        label=label,
        pop_size=pop_size,
        generations=parsed_args.generations,
        random_seed=seed,
        lj_atoms=parsed_args.atom_count,
        lj_identifier=None,
        lj_init_config=parsed_args.configuration,
        position_bounds=None,
        initial_mean=None,
        quiet=parsed_args.quiet,
        save_nth=parsed_args.save_nth,
        wales_path=wales_path,
        packmol_executable=parsed_args.packmol_executable,
        packmol_tolerance=parsed_args.packmol_tolerance,
        packmol_side_length=parsed_args.packmol_side_length,
        packmol_seed=parsed_args.packmol_seed,
    )


if __name__ == "__main__":
    args = lj_argparse()
    print(f"argparse arguments: {args}")

    bipop_rng = np.random.default_rng(seed=int(args.random_seed * 2))
    rng = np.random.default_rng(seed=args.random_seed)

    wales_path = (
        WALES_PATH
        if args.wales_path is None
        else pathlib.Path(args.wales_path)
    )

    losses = []
    # perform initial evolution
    print("Performing initial evolution.")
    label = f"{LABEL}{str(args.atom_count)}_{args.label}"
    last_mean, loss, parameters, state, ref_found = lj_evo_call(
        step_size=args.step_size,
        label=label,
        pop_size=args.pop_size,
        seed=rng.integers(1, 1000),
        parsed_args=args,
    )
    losses.append(loss)
    np.savetxt(f"bipop_lj_{args.atom_count}_losses", losses)
    bipop = bipop_init(
        default_pop_size=parameters.pop_size,
        default_step_size=args.step_size,
        random_state=bipop_rng.bit_generator.__getstate__(),
    )

    result = {
        "label": label,
        "loss": loss,
        "atom_count": args.atom_count,
        "generations": state.generation,
        "pop_size": parameters.pop_size,
        "fun_evals": state.generation * parameters.pop_size,
        "bipop": asdict(bipop),
    }
    if args.json_output is not None:
        with open(args.json_output, "a", encoding="utf-8") as f:
            f.write(f"{json.dumps(result, cls=JSONEncoder)}\n")

    # perform until 5 large restarts were run
    i = 0
    while bipop.large_restart_counter < 5 and not ref_found:
        i += 1
        pop_size, step_size, bipop = bipop_next_restart(bipop)
        label = f"{LABEL}{str(args.atom_count)}_{args.label}_r{str(i)}"
        print(
            f"Performing restart evolution {i}.\n"
            f"Population size is {pop_size}.\n"
            f"Step size is {step_size}."
        )
        last_mean, loss, parameters, state, ref_found = lj_evo_call(
            step_size=step_size,
            label=label,
            pop_size=pop_size,
            seed=rng.integers(1, 1000),
            parsed_args=args,
        )
        fun_evals = state.generation * pop_size
        losses.append(loss)
        np.savetxt(f"bipop_lj_{args.atom_count}_losses", losses)
        bipop = bipop_update(bipop=bipop, new_evals=fun_evals)
        result = {
            "label": label,
            "loss": loss,
            "atom_count": args.atom_count,
            "generations": state.generation,
            "pop_size": pop_size,
            "fun_evals": state.generation * pop_size,
            "bipop": asdict(bipop),
        }
        if args.json_output is not None:
            with open(args.json_output, "a", encoding="utf-8") as f:
                f.write(f"{json.dumps(result, cls=JSONEncoder)}\n")
