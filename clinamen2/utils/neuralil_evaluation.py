"""Functions to evaluate NeuralIL models (including ensembles)."""
import copy
import pickle
from typing import Callable, NamedTuple, Tuple

import flax
import jax
import numpy as np
import numpy.typing as npt
from neuralil.bessel_descriptors import PowerSpectrumGenerator
from neuralil.model import NeuralIL, NeuralILModelInfo


class NeuralILInputPipeline(NamedTuple):
    """Combined pipelines for input transformation.

    Args:
        neuralil_pipeline: Function pipeline performing NeuralIL specific data
            transformation.
        clinamen_pipeline: Function pipeline performing Clinamen2 specific
            data transformation.
    """

    neuralil_pipeline: Callable = None  # structure -> npy
    clinamen_pipeline: Callable = None  # dof -> structure


def create_input_pipeline(pipeline: NeuralILInputPipeline) -> Callable:
    """Create combined pipeline.

    Args:
        pipeline: Function pipeline for input transformation.
    """

    def input_pipeline(
        dof: npt.ArrayLike,
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        """Apply input pipeline."""
        transformed = copy.deepcopy(dof)

        for fun in pipeline[::-1]:
            transformed = fun(transformed) if fun is not None else transformed

        return transformed

    return input_pipeline


def create_batch_input_pipeline(pipeline: NeuralILInputPipeline) -> Callable:
    """Create combined pipeline.

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
            transformed.append(t)

        return transformed

    return input_pipeline


def load_neuralil_model(
    pickled_model: str,
    max_neighbors: int,
    core_class: type,
    model_class: type = NeuralIL,
    **kwargs,
) -> NeuralIL:
    """Load a NeuralIL model for evaluation.

    Args:
        pickled_model: Filename and full path to model.
        max_neighbors: Maximum number of neighbors to be taken into account
            for descriptor generation.
        core_class: Type of the neuralil.model core.
        model_class: Type of the neuralil.model model.
        kwargs: These parameters are passed to PowerSpectrumGenerator().

    Returns:
        tuple containing

            - Individual NeuralIL model.
            - Model info.
    """

    # unpickle the selected model
    with open(pickled_model, "rb") as f:
        model_info = pickle.load(f)

    # initialize the descriptor generator
    pipeline = PowerSpectrumGenerator(
        model_info.n_max,
        model_info.r_cut,
        len(model_info.sorted_elements),
        max_neighbors,
        **kwargs,
    )

    core_model = core_class(model_info.core_widths)
    individual_model = model_class(
        len(model_info.sorted_elements),
        model_info.embed_d,
        model_info.r_cut,
        pipeline,
        pipeline.process_some_data,
        core_model,
    )

    return individual_model, model_info


def get_ensemble_model(
    individual_model: NeuralIL, n_ensemble: int, ensemble_class: type
) -> flax.linen.Module:
    """Function to load a NeuralIL cheap ensemble for evaluation.

    Args:
        individual_model: Trained NeuralIL model.
        n_ensemble: Number of members in the ensemble.
        ensemble_class: Type of neuralil.ensemble_model

    Returns:
        NeuralIL cheap ensemble.
    """

    return ensemble_class(individual_model, n_ensemble)


def create_neuralil_calc_energy(
    dynamics_model: flax.linen.Module, model_info: NeuralILModelInfo
) -> Tuple[Callable, Callable]:
    """Closure that returns wrappers for jitted evaluation functions.

    Args:
        dynamics_model: NeuralIL model or ensemble model.
        model_info: Model info containing parameters.

    Returns:
        tuple containing

            - Closure for evaluation of single structure.
            - Closure for vmapped evaluation.
    """

    @jax.jit
    def calc_energy(positions, types, cell):
        """Calculate potential energy with a model."""
        raw = dynamics_model.apply(
            model_info.params,
            positions,
            types,
            cell,
            method=dynamics_model.calc_potential_energy,
        )

        return raw.mean()

    # vmapped version of `calc_energy`
    vmapped_calc_energy = jax.vmap(calc_energy, in_axes=(0, 0, 0))

    def calc_wrapper(inputs):
        """Wrapper for jitted `calc_energy`."""
        return calc_energy(inputs[0], inputs[1], inputs[2])

    def vmapped_calc_wrapper(inputs):
        """Wrapper for vmapped jitted `calc_energy`."""
        positions, types, cells = zip(*inputs)

        return vmapped_calc_energy(
            np.asarray(positions), np.asarray(types), np.asarray(cells)
        )

    return calc_wrapper, vmapped_calc_wrapper


def create_neuralil_calc_energy_and_forces(
    dynamics_model: flax.linen.Module, model_info: NeuralILModelInfo
) -> Tuple[Callable, Callable]:
    """Closure that returns wrappers for jitted evaluation functions.

    Args:
        dynamics_model: NeuralIL model or ensemble model.
        model_info: Model info containing parameters.

    Returns:
        tuple containing

            - Closure for evaluation of single structure.
            - Closure for vmapped evaluation.
    """

    @jax.jit
    def calc_energy_and_forces(positions, types, cell):
        """Calculate potential energy and forces with a model."""
        raw = dynamics_model.apply(
            model_info.params,
            positions,
            types,
            cell,
            method=dynamics_model.calc_potential_energy_and_forces,
        )

        return raw[0].mean(), {"raw_output": raw}

    # vmapped version of `calc_energy`
    vmapped_calc_energy_and_forces = jax.vmap(
        calc_energy_and_forces, in_axes=(0, 0, 0)
    )

    def calc_wrapper(inputs):
        """Wrapper for jitted `calc_energy_and_forces`."""
        return calc_energy_and_forces(inputs[0], inputs[1], inputs[2])

    def vmapped_calc_wrapper(inputs):
        """Wrapper for vmapped jitted `calc_energy_and_forces`."""
        positions, types, cells = zip(*inputs)
        return vmapped_calc_energy_and_forces(
            np.asarray(positions), np.asarray(types), np.asarray(cells)
        )

    return calc_wrapper, vmapped_calc_wrapper


def create_neuralil_calc_energy_and_forces_uncertainty(
    dynamics_model: flax.linen.Module, model_info: NeuralILModelInfo
) -> Tuple[Callable, Callable]:
    """Closure that returns wrappers for jitted evaluation functions.

        Also returns uncertainty calculated from forces.

    Args:
        dynamics_model: NeuralIL model or ensemble model.
        model_info: Model info containing parameters.

    Returns:
        tuple containing

            - Closure for evaluation of single structure.
            - Closure for vmapped evaluation.
    """
    (
        calc_energy_and_forces,
        vmapped_calc_energy_and_forces,
    ) = create_neuralil_calc_energy_and_forces(
        dynamics_model=dynamics_model, model_info=model_info
    )

    @jax.jit
    def calc_uncertainty_from_forces(forces):
        """Calculates an uncertainty from all forces within a structure."""
        uncertainty = forces.std(axis=0).sum(axis=(0, 1))

        return uncertainty

    def calc_wrapper(inputs):
        """Wrapper for NeuralIL evaluation.

        Ensures that the function signature is compatible with the workflow
        functions.

        Args:
            -
        """
        loss, additional = calc_energy_and_forces(inputs)
        forces = additional["raw_output"][1]
        uncertainty = calc_uncertainty_from_forces(forces)

        return loss, {
            "forces": np.asarray(forces),
            "uncertainty": np.asarray(uncertainty),
        }

    def vmapped_calc_wrapper(inputs):
        """TBD"""
        loss, additional = vmapped_calc_energy_and_forces(inputs)
        additional_output = [
            {"forces": np.asarray(fo)} for fo in additional["raw_output"][1]
        ]
        for f, forces in enumerate(additional["raw_output"][1]):
            additional_output[f]["uncertainty"] = float(
                calc_uncertainty_from_forces(forces)
            )

        return (
            loss,
            additional_output,
        )

    return calc_wrapper, vmapped_calc_wrapper
