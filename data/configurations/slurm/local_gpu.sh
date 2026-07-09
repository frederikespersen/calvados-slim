#!/bin/bash

#······································································#
# Local run settings (OpenMM OpenCL platform, e.g. Apple/local GPU)
#······································································#
# Job name: <JOB-NAME>
# Run from the simulation directory, in the background, with a walltime:
#     nohup caffeinate -i bash local_gpu.sh &
# Override the default walltime (HH:MM:SS; empty disables the limit):
#     WALLTIME=12:00:00 nohup caffeinate -i bash local_gpu.sh &
WALLTIME="${WALLTIME:-48:00:00}"

#······································································#
# Logging (mirrors the Slurm %x.out / %x.err redirection)
#······································································#

mkdir -p logs
exec > >(tee "logs/<JOB-NAME>.out") 2> >(tee "logs/<JOB-NAME>.err" >&2)

#······································································#
# Main
#······································································#

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

log "1. Setting up environment"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate calvados

log "2. Running run.py"
python run.py &
run_pid=$!

# Walltime watchdog: kill the run if it exceeds WALLTIME (mirrors Slurm -t)
if [ -n "$WALLTIME" ]; then
    IFS=: read -r h m s <<< "$WALLTIME"
    walltime_s=$((10#$h * 3600 + 10#$m * 60 + 10#$s))
    (
        sleep "$walltime_s"
        if kill -0 "$run_pid" 2>/dev/null; then
            log "Walltime $WALLTIME reached — terminating run"
            kill "$run_pid"
        fi
    ) &
    watchdog_pid=$!
fi

wait "$run_pid"
status=$?
[ -n "$watchdog_pid" ] && kill "$watchdog_pid" 2>/dev/null

log "3. Job done (exit status $status)"
exit "$status"
