"""Read and write CMA-ES results."""
import json
import pathlib
from collections import deque
from dataclasses import asdict
from typing import NamedTuple, Tuple

import dill
import numpy as np
import numpy.typing as npt

from clinamen2.cmaes.params_and_state import (
    AlgorithmParameters,
    AlgorithmState,
)
from clinamen2.cmaes.termination_criterion import Criterion


class JSONEncoder(json.JSONEncoder):
    """Class that extends JSONEncoder to handle different data types."""

    def default(self, o):
        """Return a json-izable version of o or delegate on the base class."""
        if isinstance(o, np.generic):
            # Deal with non-serializable types such as numpy.int64
            return o.item()
        elif isinstance(o, np.ndarray):
            nruter = {
                "main_type": "NumPy/" + o.dtype.name,
                "data": o.tolist(),
            }
            return nruter
        elif isinstance(o, deque):
            nruter = {
                "main_type": "deque/" + str(o.maxlen),
                "data": list(o),
            }
            return nruter
        return json.JSONEncoder.default(self, o)


class JSONDecoder(json.JSONDecoder):
    """Class that extends the JSONDecoder to handle different data types."""

    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(
            self, object_hook=self.object_hook, *args, **kwargs
        )

    def object_hook(self, obj):
        """Reencode numpy arrays from dictionary."""
        try:
            main_type, *extra = obj["main_type"].split("/")
            if main_type == "NumPy":
                return np.asarray(obj["data"], dtype=extra[0])
            elif main_type == "deque":
                maxlen = int(extra[0])
                return deque(obj["data"], maxlen=maxlen)
        except (KeyError, ValueError):
            return obj


class CMAFileHandler:
    """Handle input and output files.

    Args:
        label: Label of the evolution run.
        target_dir: Full path to target directory. Default is the current
            working directory.
    """

    def __init__(
        self,
        label: str = None,
        target_dir: pathlib.Path = pathlib.Path.cwd(),
    ):
        """Constructor

        Returns:
            Name of generated file.
        """
        self.label = label
        self.target_dir = target_dir

    def get_evolution_filename(
        self,
        label: str = None,
    ) -> pathlib.Path:
        """Return a pathlib.Path object representing the evolution file.

        Args:
            label: Label of the evolution run.
        """
        filename = "evolution"
        if label is not None:
            filename = filename + "_" + label
        elif self.label is not None:
            filename = filename + "_" + self.label
        filename = self.target_dir / filename

        return filename

    def get_generation_filename(
        self,
        generation: int,
        label: str = None,
    ) -> pathlib.Path:
        """Return a pathlib.Path object representing the generation file.

        Args:
            generation: Generation index.
            label: Label of the evolution run.
        """
        filename = "generation"
        if label is not None:
            filename = filename + "_" + label + "_" + str(generation)
        elif self.label is not None:
            filename = filename + "_" + self.label + "_" + str(generation)
        else:
            filename = filename + "_" + str(generation)
        filename = self.target_dir / filename

        return filename

    def save_evolution(
        self,
        initial_parameters: AlgorithmParameters,
        initial_state: AlgorithmState,
        termination_criterion: Criterion = None,
        label: str = None,
        additional: dict = None,
        file_format: str = "json",
        json_encoder: json.JSONEncoder = JSONEncoder,
    ) -> str:
        """Function that writes the initial evolution to file.

        Accepts any data that is serializable using dill. It might be
        necessary to use a different JSONEncoder. If 'additional' contains
        dataclasses, use `dataclasses.asdict()` and take care to manually
        re-cast after loading.

        Serialized data contains:
           | 'initial_parameters': AlgorithmParameters
           | 'initial_state': AlgorithmState
           | 'termination_criterion': Criterion (optional)
           | any additional compatible information (dill-only)

        Filenames:
            | `evolution.json`
            | `evolution_ + label + .json`

        Args:
            initial_parameters: The initial parameters to start the evolution.
            initial_state: The initial state to start the evolution.
            termination_criterion: The termination criterion set up for the
                evolution. Can be a combinated criterion.
            label: String to be added to filename.
            additional: Additional information to be saved with the
                initial evolution properties.
            file_format: Indicate the file format to be used for serialization.
                The options are 'json' and 'dill'. Default is 'json'.
            json_encoder: If default encoder file_handling.JSONEncoder does not
                offer the required functionality, e.g., JAX datatypes.
        Returns:
            Name of generated file.
        """

        filename = self.get_evolution_filename(label)

        if additional is None:
            additional = {}
        if not isinstance(additional, dict):
            raise TypeError(
                "Parameter 'additional' must be a dictionary or None."
            )

        if file_format == "json":
            if initial_parameters is not None:
                initial_parameters = asdict(initial_parameters)
            if initial_state is not None:
                initial_state = asdict(initial_state)

        data = {
            "initial_parameters": initial_parameters,
            "initial_state": initial_state,
            **additional,
        }
        if termination_criterion is not None:
            data["termination_criterion"] = termination_criterion
        if file_format == "dill":
            with open(str(filename) + ".dill", "wb") as f:
                dill.dump(data, file=f, protocol=5)
        elif file_format == "json":
            with open(
                str(filename) + ".json", "w", encoding="UTF-8"
            ) as json_file:
                json.dump(data, json_file, cls=json_encoder)
        else:
            raise NotImplementedError(
                f"Format {file_format} has not been implemented."
            )

        return filename

    def load_evolution(
        self,
        label: str = None,
        file_format: str = "json",
    ) -> Tuple[AlgorithmParameters, AlgorithmState, Criterion, dict]:
        """Function that loads an evolution from file.

        Objects loaded:
           | 'initial_parameters': AlgorithmParameters
           | 'initial_state': AlgorithmState
           | 'termination_criterion': Criterion
           | any additional compatible information

        Args:
            label: String to be added to filename.
            file_format: Indicate the file format used for serialization.
                The options are 'json' and 'dill'. Default is 'json'.

        Returns:
            tuple containing

                - AlorithmParameters object
                - AlgorithmState object
                - Criterion object
                - Dictionary containing additional objects if present.
        """
        filename = self.get_evolution_filename(label)

        if file_format == "dill":
            with open(str(filename) + ".dill", "rb") as f:
                loaded = dill.load(f)
        elif file_format == "json":
            with open(
                str(filename) + ".json", "r", encoding="UTF-8"
            ) as json_file:
                loaded = json.load(json_file, cls=JSONDecoder)
        else:
            raise ValueError(f"Format {file_format} not understood.")

        initial_parameters = loaded.pop("initial_parameters")
        initial_state = loaded.pop("initial_state")
        try:
            termination_criterion = loaded.pop("termination_criterion")
        except KeyError:
            termination_criterion = None

        if file_format == "json":
            initial_parameters = AlgorithmParameters(**initial_parameters)
            initial_state = AlgorithmState(**initial_state)

        return initial_parameters, initial_state, termination_criterion, loaded

    def save_generation(
        self,
        current_state: AlgorithmState,
        population: npt.ArrayLike,
        loss: npt.ArrayLike,
        termination_state: NamedTuple = None,
        label: str = None,
        additional: dict = None,
        file_format: str = "json",
        json_encoder: json.JSONEncoder = JSONEncoder,
    ) -> str:
        """Function that writes a generation to file.

        Accepts any data that is serializable using dill.

        Serialized data contains:
           | 'current_state': AlgorithmState
           | 'population': numpy.ArrayLike
           | 'loss': numpy.ArrayLike
           | 'termination_state': NamedTuple
           | any additional compatible information in a dictionary

        Filenames:
            'generation' + str(number) + '.json'
            'generation' + label + '_' + str(number) +'.json'

        Args:
            current_state: The current state of the evolution.
            population: The population of individuals of the generation.
            loss: Loss of each individual within the population.
            termination_state: State of termination criterion.
            label: String to be added to filename.
            additional: Additional information to be saved with the initial
                evolution properties.
            file_format: Indicate the file format to be used for serialization.
                The options are 'json' and 'dill'. Default is 'json'.
            json_encoder: If default encoder file_handling.JSONEncoder does not
                offer the required functionality, e.g., JAX datatypes.

        Returns:
            Name of generated file.
        """
        if isinstance(current_state, AlgorithmState):
            generation = current_state.generation
        elif isinstance(current_state, dict):
            generation = current_state["generation"]
        else:
            generation = 0
        filename = self.get_generation_filename(generation, label=label)

        if file_format == "json":
            if current_state is not None:
                current_state = asdict(current_state)
            if termination_state is not None:
                termination_state = termination_state._asdict()

        if additional is None:
            additional = {}
        if not isinstance(additional, dict):
            raise TypeError(
                "Parameter 'additional' must be a dictionary or None."
            )

        data = {
            "current_state": current_state,
            "population": population,
            "loss": loss,
            "termination_state": termination_state,
            **additional,
        }

        if file_format == "dill":
            with open(f"{filename}.dill", "wb") as f:
                dill.dump(data, file=f, protocol=5)
        elif file_format == "json":
            with open(f"{filename}.json", "w", encoding="UTF-8") as json_file:
                json.dump(data, json_file, cls=json_encoder)
        else:
            raise NotImplementedError(
                f"Format {file_format} has not been implemented."
            )

        return filename

    def load_generation(
        self,
        generation: int,
        label: str = None,
        allow_legacy: bool = True,
        file_format: str = "json",
    ) -> Tuple[AlgorithmState, npt.ArrayLike, npt.ArrayLike, NamedTuple, dict]:
        """Function that loads a generation from file.

        Args:
            generation: ID of generation to be loaded.
            label: String to be added to filename.
            allow_legacy: If True, older results containing 'fitness'
            instead of 'loss' can be loaded. Default is True.
            file_format: Indicate the file format used for serialization.
                The options are 'json' and 'dill'. Default is 'json'.

        Returns:
            tuple containing

                - AlgorithmState object
                - Array of population
                - Array of loss values
                - NamedTuple of termination state
                - Dictionary containing additional objects if present.
        """
        filename = self.get_generation_filename(
            generation=generation, label=label
        )

        if file_format == "dill":
            with open(str(filename) + ".dill", "rb") as f:
                loaded = dill.load(f)
        elif file_format == "json":
            with open(
                str(filename) + ".json", "r", encoding="UTF-8"
            ) as json_file:
                loaded = json.load(json_file, cls=JSONDecoder)
        else:
            raise NotImplementedError(
                f"Format {file_format} has not been implemented."
            )

        current_state = loaded.pop("current_state")
        population = loaded.pop("population")
        if file_format == "json":
            if current_state is not None:
                current_state = AlgorithmState(**current_state)
        try:
            termination_state = loaded.pop("termination_state")
        except KeyError:
            termination_state = None
        try:
            loss = loaded.pop("loss")
        except KeyError as exc:  # to be able to handle older results
            if allow_legacy:
                loss = loaded.pop("fitness")
            else:
                raise KeyError(
                    "Legacy data may not be loaded. Check settings."
                ) from exc

        return current_state, population, loss, termination_state, loaded

    def update_evolution(
        self,
        label: str = None,
        additional: dict = None,
    ) -> str:
        """Function that calls save_evolution(file_format='json').

        Accepts all data that is serializable using json.
        Values for keys that are already present in the existing additional
        information will be replaced.

        Filenames:
            | `evolution.json`
            | `evolution_ + label + .json`

        Args:
            label: Label of the evolution run.
            additional: Additional information to be saved with the
                evolution properties.

        Returns:
            Name of evolution file.
        """
        try:
            evolution = self.load_evolution(label=label)
        except FileNotFoundError:
            print("Evolution update only implemented for json format.")
            raise
        existing_additional = evolution[3] if evolution[3] is not None else {}
        updated_additional = existing_additional.copy()
        updated_additional.update(additional)
        filename = self.save_evolution(
            initial_parameters=evolution[0],
            initial_state=evolution[1],
            termination_criterion=evolution[2],
            label=label,
            additional=updated_additional,
        )
        return filename
