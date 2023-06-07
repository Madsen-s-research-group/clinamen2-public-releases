"""Example application of the BIPOP-CMA-ES.

    N. Hansen. ACM-GECCO, 2009 (inria-00382093).
    I. Loshchilov. CEC, 2013 (hal-00823880).
"""
import argparse
import json
from dataclasses import asdict
from typing import Tuple

import numpy as np
import numpy.random
from evolve_test_function import evolution

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
from clinamen2.utils.script_functions import cma_parser

LABEL = "BIPOP_"


def evo_call(
    step_size: float,
    label: str,
    pop_size: int,
    seed: int,
    parsed_args: argparse.Namespace,
) -> Tuple[str, float, AlgorithmParameters, AlgorithmState, bool]:
    """Wraps call to 'evolve_test_function.evolution()'.

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
        Result of evolution() call.
    """

    return evolution(
        function=parsed_args.function,
        dimension=parsed_args.dimension,
        step_size=step_size,
        generations=parsed_args.generations,
        label=label,
        seed=seed,
        save_nth=parsed_args.save_nth,
        output=parsed_args.output,
        mult=parsed_args.mult,
        pop_size=pop_size,
    )


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
        "-j",
        "--json_output",
        type=str,
        help="JSON file to store results in.",
        default=None,
    )
    args, unknown = parser.parse_known_args()
    print(f"argparse arguments: {args}")
    print("\n----------------------------")

    bipop_rng = np.random.default_rng(seed=int(args.random_seed * 2))
    rng = np.random.default_rng(seed=args.random_seed)

    losses = []
    # perform initial evolution
    print("Performing initial evolution.")
    label = f"{LABEL}{args.function}_{str(args.dimension)}_{args.label}"
    last_gen, loss, parameters, state = evo_call(
        step_size=args.step_size,
        label=label,
        pop_size=args.pop_size,
        seed=rng.integers(1, 1000),
        parsed_args=args,
    )
    losses.append(loss)
    np.savetxt(f"{label}_losses", losses)
    bipop = bipop_init(
        default_pop_size=parameters.pop_size,
        default_step_size=args.step_size,
        random_state=bipop_rng.bit_generator.__getstate__(),
    )

    result = {
        "label": label,
        "loss": loss,
        "function": args.function,
        "dimension": args.dimension,
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
    while bipop.large_restart_counter < 5 and loss > 1e-10:
        i += 1
        pop_size, step_size, bipop = bipop_next_restart(bipop)
        label = f"{LABEL}{str(args.dimension)}_{args.label}_r{str(i)}"
        print(
            f"Performing restart {i}.\n"
            f"Population size is {pop_size}.\n"
            f"Step size is {step_size}."
        )
        last_gen, loss, parameters, state = evo_call(
            step_size=step_size,
            label=label,
            pop_size=pop_size,
            seed=rng.integers(1, 1000),
            parsed_args=args,
        )
        fun_evals = state.generation * pop_size
        losses.append(loss)
        np.savetxt(f"{label}_losses", losses)
        bipop = bipop_update(bipop=bipop, new_evals=fun_evals)
        result = {
            "label": label,
            "loss": loss,
            "function": args.function,
            "dimension": args.dimension,
            "generations": state.generation,
            "pop_size": pop_size,
            "fun_evals": state.generation * pop_size,
            "bipop": asdict(bipop),
        }
        if args.json_output is not None:
            with open(args.json_output, "a", encoding="utf-8") as f:
                f.write(f"{json.dumps(result, cls=JSONEncoder)}\n")
