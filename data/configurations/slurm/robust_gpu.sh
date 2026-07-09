#!/bin/bash

#······································································#
# Slurm settings
#······································································#

#SBATCH --job-name=<JOB-NAME>
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=frederikespersen@icloud.com

#······································································#
# Main
#······································································#

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S:%3N')] $*"
}

log "1. Setting up environment"
source /home/fknudsen/.bashrc
conda activate calvados

export PYTHONPATH=/home/fknudsen/calvados-slim/src/calvados:$PYTHONPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

#······································································#
log "2. Running run.py"
python run.py

log "3. Job done"