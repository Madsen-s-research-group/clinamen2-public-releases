.. toctree::
   :maxdepth: 4

Getting Started
===============

This page details how to get started with Clinamen2.

Requirements
------------

Clinamen2 uses Python 3.8+ and a number of packages. We recommend
using a virtual environment, e.g., with
`Miniconda <https://conda.io/miniconda.html>`_.

.. _installation:

Installation
------------

.. highlight:: python

The source code is available at
`GitHub <https://github.com/Madsen-s-research-group/clinamen2-public-releases>`_.

To install Clinamen2 either clone the GitHub repository and call
::

  pip install -U pip
  pip install -U setuptools
  pip install -e .

or install directly from pypi.

To make sure that the installation was successful, you can install the
additional packages `pytest <https://docs.pytest.org/en/7.3.x/>`_ and
`pytest-datadir <https://pypi.org/project/pytest-datadir/>`_ manually via
::

  pip install pytest
  pip install pytest-datadir

or by installing Clinamen2 with
::

  pip install -e .[test]

to run the tests from the main directory as
::

  pytest tests

.. _tutorial:

Tutorial
--------

To demonstrate how an evolution can be set up and executed with Clinamen2, this tutorial minimizes the 8-dimensional Rosenbrock function.

First, generate an input vector (founder mean)
::

  import numpy as np

  rng = np.random.default_rng(seed=1234)
  founder = rng.random(size=8)

and prepare a function for loss evaluation, for simplicity we use the Rosenbrock implementation included in `SciyPy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.rosen.html>`_:
::

  from scipy.optimize import rosen

With this, the core algorithm can be initialized
::

  from clinamen2.cmaes.params_and_state import (
      create_sample_and_sequential_evaluate,
      create_sample_from_state,
      create_update_algorithm_state,
  )
  from clinamen2.utils.script_functions import cma_setup

  # initialize AlgorithmParameters and AlgorithmState
  parameters, initial_state = cma_setup(mean=founder, step_size=1.0)

  # The closures can be created by passing the AlgorithmParameters to the respective functions.
  update_state = create_update_algorithm_state(parameters=parameters)
  sample_individuals = create_sample_from_state(parameters)

  sample_and_evaluate = create_sample_and_sequential_evaluate(
      sample_individuals=sample_individuals,
      evaluate_loss=rosen,
  )

To run this evolution for a maximum of 1500 generations or until the loss is close to zero
::

  state = initial_state
  for g in range(1500):
      # perform one generation
      generation, state, loss = sample_and_evaluate(state=state)
      # to update the AlgorithmState pass in the sorted generation
      state = update_state(state, generation[np.argsort(loss)])

      if loss.min() < 1e-14:
          print("Evolution terminated early with success.")
          break

  # print the minimum loss in the final generation
  print(
      f"Loss {loss.min()} for individual "
      f"{loss.argmin()} in generation {g}."
  )
