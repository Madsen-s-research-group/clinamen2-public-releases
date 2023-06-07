#!/bin/bash -l
#SBATCH -J nwc-16 
#SBATCH -n 16

set -ue

module load anaconda
source activate eudaemonia
module load nwchem

export ASE_NWCHEM_COMMAND="prun nwchem PREFIX.nwi > PREFIX.nwo"
export WORKER_SCRATCH_SPACE="${LUGUS_SCRATCH_DIR}"
DASK_TEMPORARY_DIRECTORY="${WORKER_SCRATCH_SPACE}" dask-worker --nthreads 1 --nworkers 1 --local-directory "${WORKER_SCRATCH_SPACE}" --scheduler-file scheduler_nwchem.json # --worker-class distributed.Worker --no-nanny
