"""Extract evolution result and optimize locally with FIRE."""
import argparse
import pathlib
import sys

import jax
import numpy as np
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE
from neuralil.ase_integration import NeuralILASECalculator
from neuralil.committees.model import Committee
from neuralil.model import NeuralIL, ResNetCore
from packaging import version

from clinamen2.utils.file_handling import CMAFileHandler
from clinamen2.utils.neuralil_evaluation import (
    get_ensemble_model,
    load_neuralil_model,
)
from clinamen2.utils.structure_setup import prepare_dof_and_pipeline

N_ENSEMBLE = 5
PATH = pathlib.Path(pathlib.Path.cwd().parents[0] / "examples")
FOUNDER = pathlib.Path("data/si/POSCAR_bulk")

REQUIRED_JAX_VERSION = "0.4.10"

if version.parse(jax.__version__) < version.parse(REQUIRED_JAX_VERSION):
    sys.exit(
        f"Dependency issue: "
        f"JAX version needs to be at least {REQUIRED_JAX_VERSION}, "
        f"detected {jax.__version__}.\n"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    base_parser = argparse.ArgumentParser(add_help=True)
    base_parser.add_argument(
        "-l",
        "--label",
        type=str,
        default="relax",
        help="CMAES run label to optimize result of.",
    )
    base_parser.add_argument(
        "-g",
        "--generation",
        type=int,
        default=50,
        help="Generation to choose best individual from.",
    )
    base_parser.add_argument(
        "-p",
        "--poscar",
        type=str,
        default=FOUNDER,
        help="POSCAR (relative path) to relax or use as template.",
    )
    args, unknown = base_parser.parse_known_args()
    print(f"argparse arguments: {args}")

    handler = CMAFileHandler(
        label=args.label,
        target_dir=pathlib.Path.cwd() / args.label,
    )

    evo = handler.load_evolution(label=args.label)
    additional = evo[3]
    generation = args.generation
    print(f"Optimizing best individual of generation {generation}.")

    gen = handler.load_generation(generation=generation)
    loss_idx = gen[2].argmin()
    loss_dof = gen[1][loss_idx]

    founder_file = pathlib.Path.cwd() / pathlib.Path(args.poscar)

    try:
        scaled_center = additional["scaled_center"]
    except KeyError:
        scaled_center = None
    try:
        radius = additional["radius"]
    except KeyError:
        radius = None
    print(
        f"Read scaled_center {scaled_center} and "
        f"radius {radius} from evolution file."
    )
    _, transform_dof, _ = prepare_dof_and_pipeline(
        founder_file,
        scaled_center=scaled_center,
        radius=radius,
    )

    atoms_converge = transform_dof(loss_dof)

    model_path = pathlib.Path(
        pathlib.Path.cwd() / "trained_models" / "model_si_committee_51.pkl"
    )
    model, model_info = load_neuralil_model(
        pickled_model=model_path,
        max_neighbors=30,
        core_class=ResNetCore,
        model_class=NeuralIL,
    )
    ensemble = get_ensemble_model(
        individual_model=model,
        n_ensemble=N_ENSEMBLE,
        ensemble_class=Committee,
    )

    calc = NeuralILASECalculator(
        model=ensemble, model_info=model_info, max_neighbors=15
    )
    atoms_converge.calc = calc
    atoms_converge.set_pbc(True)

    # start relaxation
    label = args.label if args.label is not None else "relax"
    relax = FIRE(
        atoms=atoms_converge,
        trajectory=f"{label}_g{generation}_fire_nnff.traj",
        restart=f"{label}_g{generation}_fire_nnff.pckl",
    )
    relax.run(fmax=0.001, steps=1000)

    # read last trajectory entry and save as `POSCAR_si_optimized'
    read_traj = Trajectory(f"{label}_g{generation}_fire_nnff.traj")
    write_atoms = read_traj[-1]

    write(
        filename=f"POSCAR_{label}_g{generation}",
        images=write_atoms,
        format="vasp",
        direct=True,
        vasp5=True,
    )

    print(
        f"POSCAR of relaxed structure written to {f'POSCAR_{label}_g{generation}'}."
    )
