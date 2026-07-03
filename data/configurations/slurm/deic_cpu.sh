#!/bin/bash

#······································································#
# Slurm settings
#······································································#

#SBATCH --job-name=<JOB-NAME>
#SBATCH --partition=sbinlab_ib2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 168:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#······································································#
# Main
#······································································#

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S:%3N')] $*"
}

log "1. Setting up environment"
source /groups/sbinlab/fknudsen/.bashrc
conda activate calvados

export PYTHONPATH=/groups/sbinlab/fknudsen/calvados-slim/src/calvados:$PYTHONPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Bind OpenMM multi-threading to the allocated SLURM CPU count.
# This prevents thread oversubscription and cross-core cache thrashing.
export OPENMM_CPU_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

#······································································#
log "2. Running run.py"
python run.py

log "3. Job done"