"""Reusable code for Python script calls."""
import argparse
import pathlib
from typing import Tuple

import numpy.typing as npt

from clinamen2.cmaes.params_and_state import (
    AlgorithmParameters,
    AlgorithmState,
    create_init_algorithm_state,
    init_default_algorithm_parameters,
)
from clinamen2.utils.plot_functions import CMAPlotHelper


def cma_parser() -> argparse.ArgumentParser:
    """Argument parser for example scripts.

    Has the CMA-ES parameters as attributes.

    Further arguments added to this base parser
    may not have a shortcut.

    Returns
        parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--label",
        type=str,
        default="dummy",
        help="Run label of evolution.",
    )
    parser.add_argument(
        "-r",
        "--random_seed",
        type=int,
        default=0,
        help="Random seed for evolution. Default is 0",
    )
    parser.add_argument(
        "-s",
        "--step_size",
        type=float,
        default=1.0,
        help="Initial step size. Default is 1.0",
    )
    parser.add_argument(
        "-g",
        "--generations",
        type=int,
        default=1000,
        help="Number of generations to run for. Default is 1000",
    )
    parser.add_argument(
        "-p",
        "--pop_size",
        type=int,
        help="Population size to use instead of CMA-ES default.",
        default=None,
    )
    parser.add_argument(
        "-n",
        "--save_nth",
        type=int,
        help="Save every nth generation. Default is 1",
        default=1,
    )

    return parser


def cma_setup(
    mean,
    step_size,
    run_seed=0,
    pop_size=None,
    initial_cholesky_factor=None,
) -> Tuple[AlgorithmParameters, AlgorithmState, npt.ArrayLike]:
    """Helper function to initialize the algorithm"""
    dimension = mean.shape[0]

    # AlgorithmParameters
    parameters = init_default_algorithm_parameters(
        dimension,
        initial_step_size=step_size,
        random_seed=run_seed,
        pop_size=pop_size,
    )

    # AlgorithmState
    init_state = create_init_algorithm_state(parameters)
    state = init_state(mean=mean, cholesky_factor=initial_cholesky_factor)

    return parameters, state


def generate_result_figures(
    label: str,
    input_dir: pathlib.Path,
    generation_bounds: Tuple[int, int],
    output_dir: pathlib.Path = None,
    figsize: Tuple[float, float] = (10, 8),
    y_units: str = r"",
    json: bool = True,
) -> None:
    """Create two default figures for given evolution.

    - CMA_PlotHandler.plot_mean_loss_per_generation()
    - CMA_PlotHandler.plot_loss_per_generation()

    Args:
        label: Label of the evolution run.
        input_dir: Full path to the generated data.
        generation_bounds: Plot data between these generations.
        output_dir: Full path to save figures to. DEfaults to input_dir.
        figsize: Matpplotlib figure size. Default is (10, 8).
        y_units: Units of loss. Default is empty string.
        json: Data is stored in JSON files.

    """
    if output_dir is None:
        output_dir = input_dir

    plotter = CMAPlotHelper(
        label=label,
        generation_bounds=generation_bounds,
        input_dir=input_dir,
        output_dir=output_dir,
    )
    plotter.plot_mean_loss_per_generation(
        generation_bounds=(1, plotter.generations),
        show_sigma=True,
        sigma_e_mult=1,
        y_units=y_units,
        save_fig=True,
        figsize=figsize,
    )
    plotter.plot_loss_per_generation(
        generation_bounds=[1, plotter.generations],
        y_units=y_units,
        save_fig=True,
        figsize=figsize,
    )
