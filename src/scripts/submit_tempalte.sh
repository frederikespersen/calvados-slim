#!/bin/bash
#SBATCH --job-name=single_slim
#SBATCH --dependency=singleton
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --partition=CHANGE_ME          # EDIT: your cluster's CPU partition
#SBATCH --output=single_slim-%j.out
#SBATCH --error=single_slim-%j.err
#
# Submit the single_slim CALVADOS example (PRM + SH3 with SLiM interactions).
#
#   sbatch src/scripts/submit_single_slim.sh
#
# The run is 15 ns (1,500,000 steps at dt = 0.01 ps), one frame every 1000 steps
# (1500 frames). restart='checkpoint', so resubmitting (--dependency=singleton)
# continues from restart.chk rather than starting over.
#
# CPU threads are governed by `threads` in prepare.py's Config (default 1), which
# CALVADOS passes to OpenMM explicitly. To use N cores, set threads=N there AND
# --cpus-per-task=N below. For this 87-bead system 1 thread is plenty (and likely
# fastest); the run finishes well within the wall time. For a GPU run instead,
# set platform='CUDA' in prepare.py and swap the resource lines for
# `#SBATCH --gres=gpu:1` on a GPU partition.

set -euo pipefail

# ------------------------------- user settings -------------------------------
CONDA_ENV="calvados"                                    # conda env with calvados installed
REPO_ROOT="${CALVADOS_SLIM_ROOT:-$HOME/Repositories/calvados-slim}"
NAME="single_slim"
# -----------------------------------------------------------------------------

EXAMPLE_DIR="${REPO_ROOT}/src/calvados/examples/${NAME}"
SYSDIR="${EXAMPLE_DIR}/${NAME}"

# activate environment
source ~/.bashrc
conda activate "${CONDA_ENV}"

# use the same number of OpenMM CPU threads as cores requested
export OPENMM_CPU_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Host: $(hostname)   Cores: ${SLURM_CPUS_PER_TASK:-1}"

# prepare the system once (config.yaml, components.yaml, run.py); skip if present.
# prepare.py bakes absolute input paths, so it must run from the example dir.
cd "${EXAMPLE_DIR}"
if [[ ! -f "${SYSDIR}/config.yaml" ]]; then
    echo "Preparing system in ${SYSDIR}"
    python prepare.py --name "${NAME}"
fi

# run the simulation
python "${SYSDIR}/run.py" \
    --path "${SYSDIR}" \
    --config config.yaml \
    --components components.yaml

echo "Done."
