"""Implementation of the Runner classes to be used with Dask."""
import logging
import os
import pathlib
import pickle
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Callable, Optional

import dask.distributed
import jinja2
import numpy.typing as npt
from dask.distributed import Client, as_completed

# loss: float, information: dict
WorkerResult = namedtuple("WorkerResult", ["loss", "information"])


class Runner(ABC):
    """Abstract class to interface to any queue

    Inherit from this class to implement any loss evaluation.

    Usage:
        - instantiate
        - submit() or submit_batch(): Send structure(s) to Runner
        - pop(): Retrieve latest result from Runner

    Args:
        recreate_structure: Function that recreates a structure from a given 1D
            array.

    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def submit(self, individual: npt.ArrayLike):
        """Function to submit one structure to the Runner

        Args:
            individual: 1D array representing an individual structure.

        Returns:
            ID of input array

        """

    @abstractmethod
    def pop(self):
        """Function to fetch results from the Runner

        Returns:
            - ID of input array
            - Calculation result
        """

    def submit_batch(self, individuals: npt.ArrayLike):
        """Function that sequentially calls submit() for all individuals.

        Args:
            individuals: Input to be submitted.

        Returns:
            List of IDs of input arrays
        """
        ids = []
        for individual in individuals:
            ids.append(self.submit(individual))

        return ids


class FunctionRunner(Runner):
    """Simple Runner for local function evaluation using Dask.

    Args:
        evaluator_function: Function that calculates the loss.
            Needs to return a tuple of (float, {}).
        workers: Number of workers to be started in parallel.
        scheduler_file: Path to file identifying the dask scheduler.
        convert_input: Function that takes the input array and returns the
            object that is expected by the Runner. Default is identity.
    """

    def __init__(
        self,
        evaluator_function: Callable,
        workers: int,
        scheduler_file: Optional[pathlib.Path] = None,
        convert_input: Callable = lambda x: x,
    ):
        self.evaluator_function = evaluator_function
        client_params = {
            "threads_per_worker": 1,
            "n_workers": workers,
            "silence_logs": logging.ERROR,
        }
        if scheduler_file is not None:
            client_params["scheduler_file"] = scheduler_file
        self.dask_client = Client(**client_params)
        self.futures = as_completed([])
        self.convert_input = convert_input
        super().__init__()

    def submit(self, individual: npt.ArrayLike):
        future = self.dask_client.submit(
            self.evaluator_function, self.convert_input(individual)
        )
        self.futures.add(future)
        return future.key

    def pop(self):
        for future in self.futures:
            yield future

    def __del__(self):
        self.dask_client.close()


class ScriptRunner(Runner):
    """Runner for distributed script evaluations using Dask.

    Usage:
        This class provides the functionality to evaluate individuals in a
        distributed manner using Dask workers. The provided script is executed
        on the workers using the script_run_command in a temporary directiory.
        This directory contains the output of the convert_input function saved
        with pickle and named "input". The script is expected to write a
        pickled WorkerResult object named "result", or in case of failure save
        the pickled exception as "result".

    Args:
        script_text: Text of the script to be executed on the workers.
            Before execution the script will be rendered with jinja using the
            given script_config as context.
        script_run_command: Command line command to execute the script on the
            workers, it has to contain {SCRIPTFILE}, which will be replaced
            by the file name of the actual script. E.g "python {SCRIPTFILE}"
        script_config: Dictionary of jinja keyword - value pairs to be used for
            script rendering.
            Default is None.
        convert_input: Function that takes the input array and returns the
            object that is expected by the Runner.
            Default is identity.
        scheduler_info_path: The path to the Dask scheduler descriptor file.
            Default is None, which starts the Dask scheduler locally.
    """

    def __init__(
        self,
        script_text: str,
        script_run_command: str,
        script_config: Optional[dict] = None,
        convert_input: Callable = lambda x: x,
        scheduler_info_path: Optional[str] = None,
    ):
        env = jinja2.Environment()
        self.raw_script_text = script_text
        self.template = env.from_string(script_text)
        self.script_config = script_config
        self.script_run_command = script_run_command

        if scheduler_info_path is None:
            self.dask_client = Client()
        else:
            self.dask_client = Client(scheduler_file=scheduler_info_path)
        self.futures = as_completed([])
        self.convert_input = convert_input
        super().__init__()

    @staticmethod
    def script_driver(payload):
        script_text, script_run_command, data = payload

        worker = dask.distributed.get_worker()
        with tempfile.TemporaryDirectory(
            dir=worker.local_directory
        ) as foldername:
            with open(os.path.join(foldername, "script"), "w") as tmpscript:
                tmpscript.write(script_text)

            with open(os.path.join(foldername, "input"), "wb") as tmpdata:
                pickle.dump(data, tmpdata)

            proc = subprocess.Popen(
                script_run_command.format(
                    SCRIPTFILE=os.path.join(foldername, "script")
                ),
                shell=True,
                cwd=foldername,
            )
            while proc.poll() is None:
                time.sleep(1)

            with open(os.path.join(foldername, "result"), "rb") as resultfile:
                result = pickle.load(resultfile)

            if isinstance(result, Exception):
                raise result

            return result

    def peek_script(self, script_config=None) -> str:
        """Function to check what the jinja script rendering would result in.

        Returns:
            Rendered script text.
        """
        return self.template.render(
            self.script_config if script_config is None else script_config
        )

    def submit(self, individual, script_config=None):
        future = self.dask_client.submit(
            self.script_driver,
            (
                self.template.render(
                    self.script_config
                    if script_config is None
                    else script_config
                ),
                self.script_run_command,
                self.convert_input(individual),
            ),
        )
        self.futures.add(future)
        return future.key

    def submit_batch(self, individuals: npt.ArrayLike, script_config=None):
        ids = []
        for individual in individuals:
            ids.append(self.submit(individual, script_config))

        return ids

    def pop(self):
        for future in self.futures:
            yield future

    def __del__(self):
        try:
            self.dask_client.close()
        except AttributeError:
            pass
