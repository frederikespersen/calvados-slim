#!/bin/bash

#······································································#
# Slurm settings
#······································································#

#SBATCH --job-name=<JOB-NAME>
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=frederikespersen@icloud.com

#······································································#
# Arguments
#······································································#
# Run a whole Go-strength (k) scan on a single GPU using Nvidia MPS, which
# lets the many near-identical, GPU-underutilising CALVADOS runs share one
# GPU context and execute concurrently.
#
# Submit from a scan directory containing the k=* subdirectories (one per k
# value), each holding its own run.py:
#
#     sbatch robust_gpu_mps_scan.sh [k_min k_max k_step]
#
# With no arguments, every k=* subdirectory is run. With three arguments,
# only the k values from k_min to k_max (inclusive) in steps of k_step are
# run, matching the k={k:.2f} naming used during curation.

K_MIN=$1
K_MAX=$2
K_STEP=$3

# Number of simulations to run concurrently under MPS. Keep this <=
# --cpus-per-task: each OpenMM CUDA process has an active host driver thread,
# so oversubscribing cores erodes the MPS gain. Tune to where aggregate
# ns/day plateaus (benchmark 1, 2, 4, 8, 16 on the target GPU).
JOBS=${SLURM_CPUS_PER_TASK:-8}

# Timestamped logging: [YYYYMMDD-HHMMSSmmm] message
log() { echo "[$(date +%Y%m%d-%H%M%S%3N)] $*"; }

#······································································#
log "1. Selecting k= subdirectories"

if [ -n "$K_MIN" ] && [ -n "$K_MAX" ] && [ -n "$K_STEP" ]; then
    log "   k from $K_MIN to $K_MAX in steps of $K_STEP"
    mapfile -t k_dirs < <(
        awk -v lo="$K_MIN" -v hi="$K_MAX" -v st="$K_STEP" \
            'BEGIN { n = int((hi - lo) / st + 0.5); for (i = 0; i <= n; i++) printf "k=%.2f\n", lo + i * st }'
    )
else
    log "   all k=* subdirectories"
    mapfile -t k_dirs < <(ls -d k=*/ 2>/dev/null | sed 's:/*$::')
fi

# Keep only existing directories that hold a run.py
sims=()
for d in "${k_dirs[@]}"; do
    if [ -f "$d/run.py" ]; then
        sims+=("$d")
    else
        log "   Skipping $d (no run.py)"
    fi
done

if [ ${#sims[@]} -eq 0 ]; then
    log "   No runnable k= subdirectories found. Exiting."
    exit 1
fi
log "   ${#sims[@]} simulations selected: ${sims[*]}"

#······································································#
log "2. Setting up environment"
source /home/fknudsen/.bashrc
conda activate calvados

export PYTHONPATH=/home/fknudsen/calvados-slim/src/calvados:$PYTHONPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

#······································································#
log "3. Starting MPS daemon on the allocated GPU"

# Job-scoped, non-default socket/log paths so this MPS does not capture (or get
# captured by) other users' GPU processes on a shared node.
export CUDA_MPS_PIPE_DIRECTORY=${TMPDIR:-/tmp}/nvidia-mps-pipe-$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=${TMPDIR:-/tmp}/nvidia-mps-log-$SLURM_JOB_ID
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

# Pin the daemon to the GPU Slurm allocated to this job, then clear
# CUDA_VISIBLE_DEVICES so child runs inherit no device pin: the MPS server
# exposes the single pinned GPU as device 0, which run.py targets by default.
GPU_ID=${CUDA_VISIBLE_DEVICES:-0}
unset CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$GPU_ID nvidia-cuda-mps-control -d

# Always shut MPS down on exit (success, failure, timeout) so the GPU is
# released for other users; leaving a server running blocks the GPU.
trap 'log "Stopping MPS daemon"; echo quit | nvidia-cuda-mps-control' EXIT

#······································································#
log "4. Running ${#sims[@]} simulations, $JOBS at a time, under MPS"

# log() is not exported into parallel's subshells, so timestamp inline here.
printf '%s\n' "${sims[@]}" | parallel -j "$JOBS" 'cd {} && echo "[$(date +%Y%m%d-%H%M%S%3N)] Starting {}" && python run.py'

log "5. All simulations finished"
log "6. Job done"
