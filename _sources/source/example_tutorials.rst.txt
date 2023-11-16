Example tutorials
====================

.. highlight:: python

* :ref:`function_trial`
* :ref:`silver_cluster`
* :ref:`si_bulk`
* :ref:`lennard_jones`

.. _function\_trial:

Function trial
--------------

The function trial can be performed from the command-line or by importing the
relevant functions, e.g., to a Jupyter notebook.

From the command-line call
`evolve_test_function.py <https://github.com/Madsen-s-research-group/clinamen2-public-releases/blob/public_release_v2023.11.1/examples/evolve_test_function.py>`_
from the 'examples' directory
::

    python evolve_test_function.py --help

to see the available parameters. For example, to perform an optimization of the
Cigar function in 16 dimensions for a maximum of 1000 generations, the call is
::

    python evolve_test_function.py -l mytest -g 1000 -f cigar -d 16 -n 10 --plot_mean

Every 10th generation and the final generation are saved to disk. By also
passing in the '--plot_mean' flag, two figures are created and saved to pdf in
'examples/mytest'. Only generations that were saved to disk can be plotted.

The same result can be achieved by the following steps, with the notebook or
script placed in the 'examples' directory.
::

    import pathlib
    from evolve_test_function import evolution, generate_result_figures

    LABEL = "mytest"

    # perform the evolution and write results
    evolution(label=LABEL, function="cigar", dimension=4)

    # generate the result plots
    generate_result_figures(
        label=LABEL,
        input_dir=pathlib.Path.cwd() / LABEL,
        generation_bounds=(0, -1), # all generations,
    )

To perform multiple runs the shell script `run_function_trial.sh <https://github.com/Madsen-s-research-group/clinamen2-public-releases/blob/public_release_v2023.11.1/examples/run_function_trial.sh>`_  can be adapted.

Additionally, the function trial can be performed utilizing the BIPOP restart
by calling `evolve_test_function_bipop.py <https://github.com/Madsen-s-research-group/clinamen2-public-releases/blob/public_release_v2023.11.1/examples/evolve_test_function_bipop.py>`_
::

    evolve_test_function_bipop.py --help

For example, the Ackley function in 512 dimensions:
::

    python evolve_test_function_bipop.py -f ackley -d 512 -g 5000 -l ackley_bipop -s 12.5 -m 65.536


.. _silver\_cluster:

Silver cluster with density functional theory
---------------------------------------------

To use this example, `NWChem <https://nwchemgit.github.io/>`_ or `VASP <https://vasp.at>`_ need to be
installed on the system.

Before starting the evolution, the `Dask <https://docs.dask.org/en/stable/>`_
components need to be started directly or with the provided scripts, adapted to
your system. In preparing this example we used `tmux <https://github.com/tmux/tmux/wiki>`_
to keep the scheduler running and have it easily accessible. For consistency,
scheduler, workers and evolution need to use the same version of Dask in the
respective environments.

Depending on the choice of DFT code, different settings are required in the following steps.


To start the scheduler one can use `scheduler_start_vasp.sh <https://github.com/Madsen-s-research-group/clinamen2-public-releases/blob/public_release_v2023.11.1/examples/scheduler_start_vasp.sh>`_
or `scheduler_start_nwchem.sh <https://github.com/Madsen-s-research-group/clinamen2-public-releases/blob/public_release_v2023.11.1/examples/scheduler_start_nwchem.sh>`_,
respectively.

The direct command is

::

    dask-scheduler --port 0 --scheduler-file scheduler_<DFT>.json --interface em2 1>LOG 2>LOGERR

With this, the scheduler is active (check for example via 'ps -ef') and an associated
'scheduler_<DFT>.json' file (<DFT> is either 'vasp' or 'nwchem') has been created, with content similar to
::

    {
        "type": "Scheduler",
        "id": "Scheduler-46f55f65-8e76-4337-80d4-5906a75fb7d0",
        "address": "tcp://192.168.1.2:34475",
        "services": {
            "dashboard": 41695
        },
        "started": 1681216376.6159742,
        "workers": {}
    }

If scheduler, workers and evolution are not executed on the same system, the corresponding
scheduler json file has to be copied. It may also be necessary to set
up ssh tunnels or similar to enable communication between the components.

This example calls the DFT code through ASE (`VASP <https://wiki.fysik.dtu.dk/ase/ase/calculators/vasp.html>`__,
`NWChem <https://wiki.fysik.dtu.dk/ase/ase/calculators/nwchem.html>`__),
which requires some environment variables to be set.

- VASP: *ASE_VASP_COMMAND* and *VASP_PP_PATH*
- NWChem: *ASE_NWCHEM_COMMAND*

We additionally use a scratch directory to avoid data clutter. Using Slurm a worker
can be started as follows, executed in an Anaconda environment called 'clinamen2'.

::

    #!/bin/bash -l
    #SBATCH -J worker-16
    #SBATCH -n 16

    set -ue

    module load anaconda
    source activate clinamen2
    module load <DFT>

    # SET DFT-SPECIFIC ENVIRONMENT VARIABLES HERE

    export WORKER_SCRATCH_SPACE="${CLUSTER_SCRATCH_DIR}"
    DASK_TEMPORARY_DIRECTORY="${WORKER_SCRATCH_SPACE}" dask-worker --nthreads 1 --nworkers 1 --local-directory "${WORKER_SCRATCH_SPACE}" --scheduler-file scheduler_<DFT>.json


Like all included examples, the standard CMA-ES parameters can be passed as
command line arguments. For a description of these and the Ag cluster specific
arguments call the script `evolve_ag_cluster_dft.py <https://github.com/Madsen-s-research-group/clinamen2-public-releases/blob/public_release_v2023.11.1/examples/evolve_ag_cluster_dft.py>`_ from the 'examples' directory

::

    python evolve_ag_cluster_dft.py --help

The example-specific parameters need to be chosen with care.

* 'founder' points to an ASE compatible POSCAR with the right number of Ag atoms and a sufficiently large cell
* 'dft_backend' to choose between 'nwchem' and 'vasp'
* 'randomize_positions': If this flag is given, the atom positions read from the founder will be shuffled
* 'random_positions_limit': Defines a radius or side length to restrict the position randomization

Example call

::

    python evolve_ag_cluster_dft.py --founder data/ag/POSCAR_ag5_small_box --dft_backend nwchem -r 5 -s 1.0 --g 350 -l ag5_small_box --randomize_positions sphere



.. _si\_bulk:

Si bulk with neural-network force field
---------------------------------------

In order to use the neural-network force field (NNFF) featured in this example,
additional packages need to be installed. First of all, `Google JAX <https://github.com/google/jax>`_,
with documentation and installation guide
available at `readthedocs <https://jax.readthedocs.io/en/latest/>`_.
JAX installation instruction can also be found on the NeuralIL Github package
listed in the next paragraph. The JAX version has to be 0.4.10 or newer.

The NNFF is implemented in `NeuralIL <https://pubs.acs.org/doi/10.1021/acs.jcim.1c01380>`_
with the version compatible with Clinamen2 1.0 available at `Github <https://github.com/Madsen-s-research-group/neuralil-public-releases/tree/clinamen2>`_.
To install please follow the instructions there to install the requirements JAX
and `VeLO <https://arxiv.org/abs/2211.09760>`_ and clone the NeuralIL
public release to then install the compatible branch via
::

    git checkout clinamen2
    pip install -e .

We recommend setting memory pre-allocation to false in the command line:
::

    export XLA_PYTHON_CLIENT_PREALLOCATE=false

As per usual, the available command-line parameters can be viewed directly by
calling `evolve_si_bulk_nnff.py <https://github.com/Madsen-s-research-group/clinamen2-public-releases/blob/public_release_v2023.11.1/examples/evolve_si_bulk_nnff.py>`_ with
::

    python evolve_si_bulk_nnff.py --help

Of the example-specific parameters, 'batch' should be used when a GPU is
available and 'model' identifies the trained model for loss evaluation.
The other parameters are used to control which atoms the algorithm may
manipulate and also to optionally bias the initial covariance matrix.

* 'scaled_center': The center of the algorithm's sphere of influence. In scaled coordinates [0, 1].
* 'radius': The radius of said sphere.
* 'c_r': Biasing parameter, see manuscript for details.
* 'sigma_cov': Biasing parameter, see manuscript for details.

Example call
::

    python evolve_si_bulk_nnff.py -l sibulk -r 1 -s 0.1 -g 500 --scaled_center 0.0 0.0 0.0 --radius 4.0 --c_r 20.0

The local relaxation of any of the generations can be performed using
`optimize_si_fire_nnff.py <https://github.com/Madsen-s-research-group/clinamen2-public-releases/blob/public_release_v2023.11.1/examples/optimize_si_fire_nnff.py>`_
::

    python optimize_si_fire_nnff.py --help

Example call to optimize generation 150
::

    python optimize_si_fire_nnff.py -l sibulk -g 150


.. _lennard\_jones:

Lennard-Jones cluster
---------------------

This example optionally requires `PACKMOL <https://m3g.github.io/packmol/>`_,
as it may be used to set up founder structures. Other options are available too.

All Lennard-Jones (LJ) reference values are taken from `The Cambridge Energy Landscape Database <http://doye.chem.ox.ac.uk/abstracts/jpc97.html>`_.
These published LJ clusters can be downloaded from `doye.chem.ox.ac.uk <http://doye.chem.ox.ac.uk/jon/structures/LJ/tables.150.html>`_
and their path needs to be passed to the evolution via the 'wales_path'
parameter.

`Google JAX <https://github.com/google/jax>`_ with documentation and
installation guide available at `readthedocs <https://jax.readthedocs.io/en/latest/>`_
is required. More information can also be found in example :ref:`si_bulk`.

The Bi-Population (BIPOP) restart example is called via
`evolve_lj_cluster_jax_bipop.py <https://github.com/Madsen-s-research-group/clinamen2-public-releases/blob/public_release_v2023.11.1/examples/evolve_lj_cluster_jax_bipop.py>`_
::

    evolve_lj_cluster_jax_bipop.py --help

Parameters 'atom_count' and 'identifier' are interpreted as options presented
in the Cambridge database. The 'configuration' choice 'packmol' requires
additional parameters described in the help output. In the file given as
'json_output' information regarding the restarts will be logged.

To perform a BIPOP restart evolution of the LJ13 cluster call for example
::

    python evolve_lj_cluster_jax_bipop.py -l lj13 -s 0.25 -g 10000 -n 10 -a 13 -c sphere -j lj13_bipop.json -w <WALES_PATH>

It is also possible to perform a single LJ evolution without restarts
`evolve_lj_cluster_jax <https://github.com/Madsen-s-research-group/clinamen2-public-releases/blob/public_release_v2023.11.1/examples/evolve_lj_cluster_jax.py>`_
::

    python evolve_lj_cluster_jax.py -l lj5 -s 0.5 -g 1000 -n 10 -a 5 -c sphere -w <WALES_PATH>

Variants of the CMA-ES incorporating restarts are especially useful when investigating highly multimodal loss
landscapes. By repeatedly starting searches with different parameters, more of the focus is shifted to exploration.
The choice of restart algorithm and parameters is highly problem dependent though, with no one-size-fits-all solution.
Depending on the use case, a simple grid search across population- and step-sizes may be a good choice.
