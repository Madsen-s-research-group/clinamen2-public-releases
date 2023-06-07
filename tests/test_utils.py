"""Testcases for clinamen2.utils"""
import pathlib
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt
import pytest

from clinamen2.cmaes.cmaes_criteria import EqualFunValuesState
from clinamen2.cmaes.params_and_state import (
    AlgorithmParameters,
    AlgorithmState,
)
from clinamen2.cmaes.termination_criterion import StaleLossCriterion
from clinamen2.utils.file_handling import CMAFileHandler

pytest_plugins = ["pytest-datadir"]


@dataclass
class SomeDataclass:
    """Class to test save / load with."""

    some_int_attribute: int
    some_str_attribute: str
    some_array_attribute: npt.ArrayLike


@pytest.fixture(name="some_dataclass")
def fixture_some_dataclass() -> SomeDataclass:
    """An instance of SomeDataclass"""
    return SomeDataclass(
        1, "some", np.array([[0.5, 0.1], [1.2, 2.0], [0.01, 1.0]])
    )


@pytest.fixture(name="some_array")
def fixture_some_array() -> npt.ArrayLike:
    """A numpy array."""
    rng = np.random.default_rng(0)
    return rng.random((10, 8))


@pytest.fixture(name="some_dict")
def fixture_some_dict() -> dict:
    """A dictionary with some key-value-pairs."""
    return {"one": 1, "two": "two", "three": [1, 1, 1]}


@pytest.mark.parametrize("default", [True, False])
def test_init_cmafilehandler(shared_datadir, default) -> None:
    """Test initialization of CMAFileHandler"""
    if default:
        handler = CMAFileHandler()
        assert isinstance(handler, CMAFileHandler)
        assert handler.label is None
        assert handler.target_dir == pathlib.Path.cwd()
    else:
        handler = CMAFileHandler(
            label="test",
            target_dir=shared_datadir,
        )
        assert isinstance(handler, CMAFileHandler)
        assert handler.label == "test"
        assert handler.target_dir == shared_datadir


@pytest.mark.parametrize("label", [None, "test"])
@pytest.mark.parametrize("file_format", ["dill", "json"])
def test_save_evolution(
    minimal_parameters, minimal_state, tmp_path, label, file_format
) -> None:
    """Test saving of initial evolution."""
    handler = CMAFileHandler(target_dir=tmp_path)
    func = handler.save_evolution
    func(
        minimal_parameters, minimal_state, label=label, file_format=file_format
    )

    if label is None:
        filename = "evolution." + file_format
    else:
        filename = "evolution_" + label + "." + file_format
    assert (tmp_path / filename).exists()


@pytest.mark.parametrize("label", [None, "test"])
def test_save_evolution_with_criterion(
    minimal_parameters, minimal_state, tmp_path, label
) -> None:
    """Test saving of initial evolution."""
    criterion = StaleLossCriterion(
        parameters=minimal_parameters, threshold=1e-5, generations=10
    )
    file_format = "dill"  # criterion not serializable with json
    handler = CMAFileHandler(target_dir=tmp_path)
    handler.save_evolution(
        initial_parameters=minimal_parameters,
        initial_state=minimal_state,
        termination_criterion=criterion,
        label=label,
        file_format=file_format,
    )
    if label is None:
        filename = f"evolution.{file_format}"
    else:
        filename = f"evolution_{label}.{file_format}"
    assert (tmp_path / filename).exists()


@pytest.mark.parametrize("file_format", ["dill", "json"])
def test_save_evolution_with_additional(
    some_dataclass,
    some_array,
    some_dict,
    minimal_parameters,
    minimal_state,
    tmp_path,
    file_format,
) -> None:
    """Test saving of initial evolution with additional information."""
    handler = CMAFileHandler(target_dir=tmp_path, label="with_additional")
    if file_format == "dill":
        additional = {
            "some_dataclass": some_dataclass,
            "some_array": some_array,
            "some_dict": some_dict,
        }
    elif file_format == "json":
        additional = {
            "some_dataclass": asdict(some_dataclass),
            "some_array": some_array,
            "some_dict": some_dict,
        }
    handler.save_evolution(
        minimal_parameters,
        minimal_state,
        additional=additional,
        file_format=file_format,
    )
    assert pathlib.Path(
        str(handler.get_evolution_filename()) + "." + file_format
    ).exists()


@pytest.mark.parametrize("label", [None, "test", "with_criterion"])
@pytest.mark.parametrize("file_format", ["dill", "json"])
def test_load_evolution(shared_datadir, label, file_format) -> None:
    """Test loading of initial evolution."""
    handler = CMAFileHandler(target_dir=shared_datadir)
    if file_format == "dill" or (
        file_format == "json" and label != "with_criterion"
    ):
        parameters, state, _, _ = handler.load_evolution(
            label=label, file_format=file_format
        )
        assert isinstance(parameters, AlgorithmParameters)
        assert isinstance(state, AlgorithmState)


@pytest.mark.parametrize("file_format", ["dill", "json"])
def test_load_evolution_with_additional(
    shared_datadir, some_dataclass, some_array, some_dict, file_format
) -> None:
    """Test loading with automatic splitting of the data."""
    label = "with_additional"
    handler = CMAFileHandler(target_dir=shared_datadir, label=label)
    parameters, state, _, additional = handler.load_evolution(
        file_format=file_format
    )

    assert isinstance(parameters, AlgorithmParameters)
    assert isinstance(state, AlgorithmState)
    assert isinstance(additional, dict)

    read_array = additional["some_array"]
    read_dict = additional["some_dict"]
    if file_format == "dill":
        read_dataclass = additional["some_dataclass"]
        assert (
            read_dataclass.some_int_attribute
            == some_dataclass.some_int_attribute
        )
        assert np.all(
            read_dataclass.some_array_attribute
            == some_dataclass.some_array_attribute
        )
        assert (
            read_dataclass.some_str_attribute
            == some_dataclass.some_str_attribute
        )
    assert np.all(read_array == some_array)
    assert read_dict == some_dict


@pytest.mark.parametrize("label", [None, "test"])
@pytest.mark.parametrize("file_format", ["dill", "json"])
def test_save_generation(minimal_state, tmp_path, label, file_format) -> None:
    """Test saving of generation."""
    rng = np.random.default_rng(0)
    population = rng.random((10, 8))
    loss = rng.random(10)
    handler = CMAFileHandler(target_dir=tmp_path)
    handler.save_generation(
        minimal_state, population, loss, label=label, file_format=file_format
    )
    filename = pathlib.Path(
        f"{handler.get_generation_filename(generation=0, label=label)}.{file_format}"
    )
    assert filename.exists()


@pytest.mark.parametrize("file_format", ["dill", "json"])
def test_save_generation_with_termination(
    minimal_state, tmp_path, file_format
) -> None:
    """Test saving of generation."""
    label = "with_termination"
    rng = np.random.default_rng(0)
    population = rng.random((10, 8))
    loss = rng.random(10)
    termination_state = EqualFunValuesState(
        fun_values=deque([1, 2, 3], maxlen=10)
    )
    handler = CMAFileHandler(target_dir=tmp_path)
    handler.save_generation(
        current_state=minimal_state,
        population=population,
        loss=loss,
        termination_state=termination_state,
        label=label,
        file_format=file_format,
    )
    filename = pathlib.Path(
        f"{handler.get_generation_filename(generation=0, label=label)}.{file_format}"
    )
    assert filename.exists()


@pytest.mark.parametrize("file_format", ["dill", "json"])
def test_save_generation_with_additional(
    minimal_state, tmp_path, some_dataclass, some_array, some_dict, file_format
) -> None:
    """Test saving of initial evolution."""
    label = "with_additional"
    additional = {
        "some_array": some_array,
        "some_dict": some_dict,
    }
    if file_format == "dill":
        additional["some_dataclass"] = some_dataclass
    rng = np.random.default_rng(0)
    population = rng.random((10, 8))
    loss = rng.random(10)
    handler = CMAFileHandler(target_dir=pathlib.Path(tmp_path))
    handler.save_generation(
        minimal_state,
        population,
        loss,
        label=label,
        additional=additional,
        file_format=file_format,
    )
    filename = pathlib.Path(
        f"{handler.get_generation_filename(generation=0, label=label)}.{file_format}"
    )
    assert filename.exists()


@pytest.mark.parametrize("label", [None, "test", "with_termination"])
@pytest.mark.parametrize("file_format", ["dill", "json"])
def test_load_generation(shared_datadir, label, file_format) -> None:
    """Test loading of a generation."""
    rng = np.random.default_rng(0)
    population = rng.random((10, 8))
    loss = rng.random(10)
    handler = CMAFileHandler(target_dir=shared_datadir)
    (
        state,
        loaded_population,
        loaded_loss,
        termination_state,
        _,
    ) = handler.load_generation(
        generation=0, label=label, file_format=file_format
    )
    if file_format == "json":
        if label == "with_termination":
            termination_state = EqualFunValuesState(**termination_state)
    assert isinstance(state, AlgorithmState)
    assert np.all(population == loaded_population)
    assert np.all(loss == loaded_loss)
    if label == "with_termination":
        assert isinstance(termination_state, EqualFunValuesState)
        assert isinstance(termination_state.fun_values, deque)
        assert termination_state.fun_values.maxlen == 10


@pytest.mark.parametrize("file_format", ["dill", "json"])
def test_load_generation_with_additional(
    shared_datadir, some_dataclass, some_array, some_dict, file_format
) -> None:
    """Test loading of a generation with additional info."""
    label = "with_additional"
    rng = np.random.default_rng(0)
    population = rng.random((10, 8))
    loss = rng.random(10)
    handler = CMAFileHandler(target_dir=shared_datadir)
    (
        state,
        loaded_population,
        loaded_loss,
        _,
        additional,
    ) = handler.load_generation(
        generation=0, label=label, file_format=file_format
    )

    assert isinstance(state, AlgorithmState)
    assert np.all(population == loaded_population)
    assert np.all(loss == loaded_loss)

    read_array = additional["some_array"]
    read_dict = additional["some_dict"]
    if file_format == "dill":
        read_dataclass = additional["some_dataclass"]
        assert (
            read_dataclass.some_int_attribute
            == some_dataclass.some_int_attribute
        )
        assert np.all(
            read_dataclass.some_array_attribute
            == some_dataclass.some_array_attribute
        )
        assert (
            read_dataclass.some_str_attribute
            == some_dataclass.some_str_attribute
        )
    assert np.all(read_array == some_array)
    assert read_dict == some_dict
