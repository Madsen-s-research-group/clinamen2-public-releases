#!/bin/bash -l
#SBATCH -J vasp-16
#SBATCH -n 16

set -ue

module load anaconda openblas scalapack vasp/6
source activate eudaemonia

export ASE_VASP_COMMAND="prun vasp_std"
export VASP_PP_PATH="/home/fbuchner/ase_vasp/__ase_vasp_pps"
export WORKER_SCRATCH_SPACE="${LUGUS_SCRATCH_DIR}"
DASK_TEMPORARY_DIRECTORY="${WORKER_SCRATCH_SPACE}" dask-worker --nthreads 1 --nworkers 1 --local-directory "${WORKER_SCRATCH_SPACE}" --scheduler-file scheduler_vasp.json # --worker-class distributed.Worker --no-nanny
