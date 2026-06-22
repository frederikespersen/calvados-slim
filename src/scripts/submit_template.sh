#!/bin/bash

#······································································#
# SUBMISSION TEMPLATE
# --------------------------
# Template for a submitting a single topology in a pdb file for simulation
# Fill out fields marked by '<field>', and place the file in the corresponding ~/exp/< >/ folder.
#······································································#

#······································································#
# Slurm settings
#······································································#

#SBATCH --partition=<PARTITION>
#SBATCH --job-name=<JOB_NAME>
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH --output=results/<JOB_NAME>.out
#SBATCH --error=results/<JOB_NAME>.err

#······································································#
# Main
#······································································#

echo "1. Setting up environment"
source /home/fknudsen/.bashrc
conda activate calvados
export PYTHONPATH=/home/fknudsen/calvados-slim/src/calvados:$PYTHONPATH

#······································································#
echo "2. Navigating to simulation directory"
cd /home/fknudsen/calvados-slim/data/calvados/single_slim_example

#······································································#
echo "3. Running prepare.py"
python prepare.py --name sh3-prm
