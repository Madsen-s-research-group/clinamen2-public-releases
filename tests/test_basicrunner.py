import dask
import numpy as np
import numpy.random
import numpy.typing as npt
import pytest
from scipy.optimize import rosen

from clinamen2.runner.basic_runner import FunctionRunner, Runner, ScriptRunner

pytest_plugins = ["pytest-datadir"]


@pytest.fixture(name="dask_setup", autouse=True)
def fixture_dask_setup(shared_datadir) -> None:
    dask.config.set(
        {
            "distributed.worker.use-file-locking": False,
            "temporary-directory": shared_datadir,
        }
    )


class RunnerTest(Runner):
    """Inherit from Runner to test submit_batch() function"""

    rng: np.random.Generator  # very basic, for test only

    def __init__(self):
        self.rng = np.random.default_rng(seed=0)
        super().__init__()

    def submit(self, individual: npt.ArrayLike):
        """Function to submit one structure to the Runner

        Args:
            individual: 1D array representing an individual structure.

            Returns:
                ID of input array

        """
        return self.rng.random()

    def pop(self):
        """Cannot be abstract"""
        pass


@pytest.fixture(name="zeros_individuals")
def fixture_zeros_individuals() -> npt.ArrayLike:
    """Fixture creating a population of individuals with all zeros."""

    return np.zeros(shape=(100, 10))


@pytest.fixture(name="test_runner")
def fixture_test_runner() -> RunnerTest:
    trunner = RunnerTest()

    return trunner


@pytest.fixture(name="function_runner")
def fixture_function_runner() -> FunctionRunner:
    rosen_tuple = lambda x: (rosen(x), {})
    function_runner = FunctionRunner(rosen_tuple, 1)

    return function_runner


def test_init_runner_default(test_runner) -> None:
    """Test constructor with default values"""

    assert isinstance(test_runner, Runner)


def test_submit_batch(test_runner, zeros_individuals) -> None:
    """Test submit_batch() method"""

    result = test_runner.submit_batch(zeros_individuals)

    assert len(result) == 100


def test_init_funcrunner(function_runner) -> None:
    """Test initialization of FunctionRunner"""

    assert isinstance(function_runner, FunctionRunner)


def test_submit_funcrunner(function_runner) -> None:
    """Test FunctionRunner.submit()"""

    key = function_runner.submit(np.asarray([1.0, 1.0]))
    print(key)
    assert str(key).startswith("lambda")
    for _ in function_runner.pop():
        pass


def test_submit_batch_funcrunner(function_runner) -> None:
    """Test FunctionRunner.submit_batch()"""

    generator = np.random.default_rng(seed=0)
    keys_list = function_runner.submit_batch(generator.uniform(size=(5, 2)))

    assert len(keys_list) == 5
    for _ in function_runner.pop():
        pass


def test_pop_funcrunner(function_runner) -> None:
    """Test FunctionRunner.pop()"""
    generator = np.random.default_rng(seed=0)
    keys_list = function_runner.submit_batch(generator.uniform(size=(100, 2)))

    returned_keys_list = []
    for future in function_runner.pop():
        returned_keys_list.append(future.key)

    keys_list.sort()
    returned_keys_list.sort()

    assert returned_keys_list == keys_list


def test_destructor_funcrunner(function_runner) -> None:
    """Test destructor of FunctionRunner"""
    assert type(function_runner.dask_client.profile()) is dict
    function_runner.__del__()

    with pytest.raises(RuntimeError):
        function_runner.dask_client.profile()


@pytest.fixture(name="script_runner")
def fixture_script_runner(shared_datadir) -> ScriptRunner:
    with open(shared_datadir / "script_runner_testscript.py.j2", "r") as f:
        script_text = f.read()
    script_run_command = "python {SCRIPTFILE}"
    script_config = {"denominator": "3"}
    script_runner = ScriptRunner(
        script_text=script_text,
        script_config=script_config,
        script_run_command=script_run_command,
    )

    return script_runner


@pytest.fixture(name="script_runner_error")
def fixture_script_runner_error(shared_datadir) -> ScriptRunner:
    with open(shared_datadir / "script_runner_testscript.py.j2", "r") as f:
        script_text = f.read()
    script_run_command = "python {SCRIPTFILE}"
    script_config = {"denominator": "0"}
    script_runner_error = ScriptRunner(
        script_text=script_text,
        script_config=script_config,
        script_run_command=script_run_command,
    )

    return script_runner_error


def test_submit_scriptrunner(script_runner) -> None:
    """Test ScriptRunner.submit()"""

    key = script_runner.submit(np.asarray([1.0, 1.0]))

    for future in script_runner.pop():
        pass

    assert str(key).startswith("script_driver")
