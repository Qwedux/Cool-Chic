#!/bin/bash
#SBATCH --mail-type=NONE # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/itet-stor/jparada/net_scratch/Cool-Chic/coolchic/jobs/%j.out # where to store the output (%j is the JOBID), subdirectory "jobs" must exist
#SBATCH --error=/itet-stor/jparada/net_scratch/Cool-Chic/coolchic/jobs/%j.err # where to store error messages
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --exclude=hardin01,tikgpu[02-10]
#SBATCH --array=0-24%8
#CommentSBATCH --nodelist=tikgpu01 # Specify that it should run on this particular node
#CommentSBATCH --account=tik-internal
#CommentSBATCH --constraint='titan_rtx|tesla_v100|titan_xp|a100_80gb'

ETH_USERNAME=jparada
PROJECT_NAME=Cool-Chic/coolchic
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}
CONDA_ENVIRONMENT=cool_chic_venv
mkdir -p ${DIRECTORY}/jobs
#TODO: change your ETH USERNAME and other stuff from above according + in the #SBATCH output and error the path needs to be double checked!

# Exit on errors
set -o errexit

# # Set a directory for temporary files unique to the job with automatic removal at job termination
# TMPDIR=$(mktemp -d)
# if [[ ! -d ${TMPDIR} ]]; then
# echo 'Failed to create temp directory' >&2
# exit 1
# fi
# trap "exit 1" HUP INT TERM
# trap 'rm -rf "${TMPDIR}"' EXIT
# export TMPDIR

# # Change the current directory to the location where you want to store temporary files, exit if changing didn't succeed.
# # Adapt this to your personal preference
# cd "${TMPDIR}" || exit 1

# Send some noteworthy information to the output log

echo "Running on node: $(hostname)"
echo "In directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
echo "Conda activated"
cd ${DIRECTORY}

# Execute your code
python3 lossless_encode.py $SLURM_ARRAY_TASK_ID RGB false
python3 lossless_encode.py $SLURM_ARRAY_TASK_ID YCoCg false

# Send more noteworthy information to the output log
echo "Finished at: $(date)"

# End the script with exit code 0
exit 0
