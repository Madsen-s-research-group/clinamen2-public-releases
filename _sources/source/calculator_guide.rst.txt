Calculator guide
====================

.. highlight:: python

* :ref:`basics`
* :ref:`function_wrapper`
* :ref:`script_runner`


.. _basics:

Basics
------

In principle, any ASE calculator can be utilized as a loss calculation backend to drive an evolution.
For a simple use case this may be sequential evaluation from a wrapper function calling the calculator.
For parallel calls to DFT the
`ScriptRunner <https://github.com/Madsen-s-research-group/clinamen2-public-releases/blob/public_release_v2023.11.1/clinamen2/runner/basic_runner.py>`_
class on top of the Dask framework can be employed.


.. _function\_wrapper:

Function wrapper
----------------

To show how to wrap an ASE calculator in a function we perform an evolution of an LJ5 cluster with
`ase.calculators.lj.LennardJones <https://wiki.fysik.dtu.dk/ase/ase/calculators/others.html>`_ as the backend.
The full example can be found in the notebook
`calculator_tutorial.ipynb <https://github.com/Madsen-s-research-group/clinamen2-public-releases/blob/public_release_v2023.11.1/examples/calculator_tutorial.ipynb>`_

Here, we illustrate the key steps to be taken when utilizing a calculator.

1. Get the degrees of freedom from an Atoms object. In this case, the coordinates of the first atom are fixed.

::

    def dof_from_atoms(atoms):
        """flatten the positions into an array"""

        # coordinates of the first atom remain fixed
        return atoms.get_positions().flatten()[3:]


    founder = dof_from_atoms(founder_atoms)

2. Instantiate a closure that can be passed to the algorithm for loss evaluation

This needs to create an atoms object and call the energy calculation (via the calculator object).

::

    import numpy as np

    from ase.calculators.lj import LennardJones

    # we need a simple closure to
    #  - translate between the CMA and the ASE calculator
    #  - calculate the loss (LJ energy)
    def get_eval_closure(founder_atoms):
        """Return a closure for transformation and evaluation"""

        calc = LennardJones()

        def evaluate_dof(dof):
            """return the LJ energy"""

            atoms = founder_atoms.copy()
            atoms.positions[1:] = dof.reshape((-1, 3))  # 1st atom fixed
            atoms.set_calculator(calc)

            energy = atoms.get_potential_energy()

            return energy

        return evaluate_dof


    eval_closure = get_eval_closure(founder_atoms)

With this, the evolution can be set up and run (see
`calculator_tutorial.ipynb <https://github.com/Madsen-s-research-group/clinamen2-public-releases/blob/public_release_v2023.11.1/examples/calculator_tutorial.ipynb>`_).


.. _script\_runner:

ScriptRunner
------------

To utilize a different DFT code through an ASE calculator a couple of scripts need to be created
and / or adapted. This guide aims to illustrate the steps to get from an NWChem call to a Vasp call.

1. Create a Jinja2 template for the actual call

Copying `nwchem_script.py.j2 <https://github.com/Madsen-s-research-group/clinamen2-public-releases/blob/public_release_v2023.11.1/examples/runner_scripts/nwchem_script.py.j2>`_,
first change the calculator import to
::

    from ase.calculators.vasp import Vasp

We use Jinja2 to parse and replace input at runtime. The parsing can remain unchanged, but the Vasp
calculator needs to be instantiated instead.

::

    atoms.calc = Vasp(
    {%- for key, value in vasp_params.items() %}
        {{key}}={{value}},
    {%- endfor %}
    )

The calculation result has to be serialized as a
`WorkerResult <https://github.com/Madsen-s-research-group/clinamen2-public-releases/blob/public_release_v2023.11.1/clinamen2/runner/basic_runner.py>`_
where the "information" dictionary may contain any data that is serializable.

In the NWChem example we chose to simplify the calculation result to a SinglePointCalculator to demonstrate the possibility.
This needs to be done if the original calculator is not serializable, e.g., due to MPI code as in the GPAW calculator. The Vasp
calculator can be serialized as-is though, such that the conversion to a SinglePointCalculator can be left out.

2. Adapt or recreate the evolution script

All necessary calculation parameters are passed at runtime as a dictionary and parsed into the Jinja2 template.
The parameters need to be structured in the way the specific calculator expects them to be.
In the Vasp Ag example, these parameters may be as shown below. Make sure to use the newly create Jinja2 template.

::

    SCRIPT_CONFIG = {
        "vasp_params": {
            "nsw": 0,
            "gga": "'PE'",
            "pp": "'PBE'",
            "ispin": 2,
            "isym": 0,
            "ismear": 0,
            "sigma": 0.0001,
            "ediff": 1e-6,
            "nelm": 80,
            "kpts": (1, 1, 1),
            "lorbit": 11,
            "lcharg": False,
            "lwave": False,
            "ncore": 8,
        },
    }

    with open(
        pathlib.Path.cwd() / "runner_scripts" / "nwchem_script.py.j2",
        "r",
        encoding="utf-8",
    ) as f:
        SCRIPT_TEXT = f.read()

Additionally, make sure to use a suitable name for the scheduler file, e.g.,

::

    runner = ScriptRunner(
        script_text=SCRIPT_TEXT,
        script_config=SCRIPT_CONFIG,
        script_run_command="python {SCRIPTFILE}",
        convert_input=transform_dof,
        scheduler_info_path="scheduler_vasp.json",
    )

3. Scheduler and workers

The scheduler and worker starts need to reflect this choice of filename

::

    dask-scheduler --port 0 --scheduler-file scheduler_vasp.json --interface em2 1>LOG 2>LOGERR

Make sure to set all the required system variables in the worker start script, if not already set.
(Commented out in the example below as a reminder.)

::

    #!/bin/bash -l
    #SBATCH -J vasp-16
    #SBATCH -n 16

    set -ue
    module load anaconda
    source activate clinamen2
    module load vasp

    # export ASE_VASP_COMMAND="..."
    # export VASP_PP_PATH="..."

    export WORKER_SCRATCH_SPACE="${CLUSTER_SCRATCH_DIR}"
    DASK_TEMPORARY_DIRECTORY="${WORKER_SCRATCH_SPACE}" dask-worker --nthreads 1 --nworkers 1 --local-directory "${WORKER_SCRATCH_SPACE}" --scheduler-file scheduler_vasp.json
