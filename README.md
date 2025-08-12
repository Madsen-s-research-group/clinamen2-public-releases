# Clinamen2

**Clinamen2** is a Python implementation of the CMA-ES algorithm designed with an emphasis on modularity and flexibility. It can be considered a spiritual successor of [Clinamen](https://gitlab.com/Marrigoni/clinamen/) written from scratch.

## General description

This repository contains releases of the software package described in the manuscript *Clinamen2: Functional-style evolutionary optimization in Python for atomistic structure searches* by Ralf Wanzenböck, Florian Buchner, Péter Kovács, Georg K. H. Madsen and Jesús Carrete at the Institute of Materials Chemistry of TU Wien.

## Quick start

Example use cases are provided as scripts in the `examples/` subdirectory. For details on how to set up and run the examples please see the [GitHub pages](https://madsen-s-research-group.github.io/clinamen2-public-releases/source/getting_started.html).

## Installation

It is recommended to install the code to a virtual environment, e.g., `conda`.
Most example applications require additional software to be installed
separately. Please see the detailed instructions in the documentation that
also contains links to the required packages.

To use `Clinamen2` with the provided test function example or to build your
own application, it can be installed via
```bash
    pip install -U pip
    pip install -U setuptools
    pip install -e .[test]
```
The `[test]` flag is not mandatory, but recommended to then be able to
verify the installation by running the tests from the main directory.
```bash
    pytest tests
```

## Example execution

To execute the test function example, switch to the `examples` directory and
run an evolution on the four dimensional sphere function.
```bash
    python evolve_test_function.py -l testrun
```
The results (evolution and generation data) is written to `examples/testrun`.

For further parameters see the documentation or call
```bash
    python evolve_test_function.py --help
```

## Package structure

The main directory contains sub folders for the core algorithm (`clinamen2`),
evolution results of test functions (`datasets`), the documentation (`docs`),
example applications (`examples`) that are described in the manuscript and
documentation and test cases (`tests`).

```bash
.
├── AUTHORS
├── clinamen2
    # The implementation of the CMA-ES algorithm and adjacent code, for
    # example, utilities.
│   ├── cmaes
│   │   ├── cmaes_criteria.py
            # This file contains a selection of standard termination criteria.
│   │   ├── params_and_state.py
            # The implementation of the core algorithm: The dataclasses for
            # parameters and state of the evolution, all update functions
            # for the CMA-ES and functions to create closures combining
            # sampling and evaluation.
│   │   └── termination_criterion.py
            # The code used for implementing termination criteria and
            # combinations (OR and AND).
│   ├── runner
│   │   ├── basic_runner.py
            # This file contains the Dask specific implemenation of Runner
            # classes that allow for example the execution of external scripts
            # on dedicated workers.
│   ├── test_functions.py
        # Functions to create closures for evaluation of the test functions
        # used in the function trial example of the manuscript, e.g., sphere
        # and cigar.
│   └── utils
│       ├── bipop_restart.py
            # This file contains the dataclass and functions for utilizing the
            # Bi-Population restart scheme.
│       ├── file_handling.py
            # The CMAFileHandler class that offers functions for saving and
            # loading of evolution and generation data, including any
            # json-serializable data.
│       ├── jax_data.py
            # This file contains a JSONEncoder that can handle JAX arrays.
│       ├── lennard_jones.py
            # A collection of functions used in the Lennard-Jones example
            # described in the manuscript.
│       ├── neuralil_evaluation.py
            # This file contains code that enables the utilization of
            # compatible NeuralIL models for loss evaluation.
│       ├── plot_functions.py
            # The CMAPlotHelper class that implements functions to be used for
            # illustrating evolution trajectories and adjacent data.
            # This encapsulates calls to the CMAFileHandler.
│       ├── script_functions.py
            # Some reusable functions for applications, e.g., to parse the
            # default CMA-ES parameters from the command line.
│       ├── structure_setup.py
            # A collection of classes and functions for the handling of
            # atomic structures, including for example the extraction of
            # degrees of freedom and rebuilding of atoms objects from those.
│       └── symmetry_setup.py
            # Extends the structure_setup.py module by implementing symmetry 
            # operations such as rotations and reflections for the reduction
            # of the number of degrees of freedom during the evolution.
├── datasets
    # This directory contains results of function trial evolution runs.
├── docs
    # The documentation sources. To recompile locally additional requirements
    # need to be met, e.g., JAX needs to be installed for linkage.
├── examples
    # This package contains the example applications detailed in the
    # manuscript.
│   ├── data
        # Data that is used in the different example applications.
│   │   ├── ag
            # POSCARs for the 5-, 6- and 7-atom Ag clusters.
│   │   ├── mos2
            # The POSCAR used as the founder for the MoS2 evolution.
│   │   └── si
            # The POSCAR used as the founder for the Si evolution.
│   ├── evolve_ag_cluster_dft.py
        # Silver cluster with density functional theory
        # This example performs an evolution of an Ag cluster. Be aware that
        # a scheduler and workers that run NWChem or VASP need to be started.
│   ├── evolve_lj_cluster_jax_bipop.py
        # An evolution of Lennard-Jones clusters within the Bi-Population
        # restart scheme.
│   ├── evolve_lj_cluster_jax.py
        # A single evolution run of a Lennard-Jones cluster. The functions
        # implemented here are used in the _bipop example.
│   ├── evolve_mos2_mace.py
        # An evolution of monolayer MoS2 utilizing a MACE committee and 
        # additional symmetry constraints.
│   ├── evolve_si_bulk_nnff.py
        # An evolution of Si bulk utilizing a NeuralIL surrogate model.
│   ├── evolve_test_function_bipop.py
        # A BIPOP scheme utilization used on test functions.
│   ├── evolve_test_function.py
        # The script used for test function evolution. Utilized in the _bipop
        # variation.
│   ├── example_function_trial.ipynb
        # This Jupyter notebook illustrates how the functions implemented in
        # the function trial example can be called directly.
│   ├── optimize_si_fire_nnff.py
        # This script contains the local relaxation of Si bulk structures via
        # ASE and NeuralIL.
│   ├── run_function_trial.sh
        # A shell script that calls a test function for a given array of
        # dimensions and different initial random seeds.
│   ├── runner_scripts
        # This package contains the jinja2 scripts that interface the
        # ScriptRunner (Dask) to specific DFT codes.
│   │   ├── gpaw_script.py.j2
│   │   ├── nwchem_script.py.j2
│   │   └── vasp_script.py.j2
│   ├── scheduler_start_nwchem.sh
        # Shell script to start a Dask scheduler for the Ag example using NWChem.
│   ├── scheduler_start_vasp.sh
        # Shell script to start a Dask scheduler for the Ag example using VASP.
│   ├── trained_models
        # Contains the NeuralIL model that is used in the Si bulk example.
│   │   └── mos2_mace_committee
            # Contains the MACE committee that is used in the MoS2 example.
│   └── worker_start_nwchem.sh
        # Shell script to submit a Dask worker to Slurm for the Ag example using NWChem.
│   └── worker_start_vasp.sh
        # Shell script to submit a Dask worker to Slurm for the AG example using VASP.
├── LICENSE
├── NOTICE
├── pyproject.toml
├── README.md
    # This file.
├── tests
    # This package contains the pytest tests that can be run to verify the
    # validity of the installation.
│   ├── data
        # Data to test against.
```
