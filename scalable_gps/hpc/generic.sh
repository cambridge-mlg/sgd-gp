#!/bin/bash

# adapted from https://github.com/y0ast/slurm-for-ml/blob/master/generic.sh
# and the Cambridge HPC sample submission script at /usr/local/Cluster-Docs/SLURM

# This is a generic running script. It can run in two configurations:
# Single job mode: pass the python arguments to this script
# Batch job mode: pass a file with first the job tag and second the commands per line

#! Which project should be charged (NB Wilkes2 projects end in '-GPU'):
#SBATCH -A T2-CS117-GPU

#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#SBATCH --ntasks=1

#! Send mail since queue times are long
#SBATCH --mail-type=FAIL

#! Do not change:
#SBATCH -p pascal

#! Specify the number of GPUs per node (between 1 and 4; must be 4 if nodes>1).
#! Note that the job submission script will enforce no more than 3 cpus per GPU
#!SBATCH --cpus-per-task=3
#!SBATCH --gres=gpu:1

#! Estimated maximum memory needed (job is force-stopped if exceeded):
#! RAM is allocated in ~5980mb blocks, you are charged per block used,
#! and unused fractions of blocks will not be usable by others.
#SBATCH --mem=20gb

#! sbatch directives end here (put any additional directives above this line)

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel7/default-gpu              # REQUIRED - loads the basic environment

# My custom modules
source ~/.bashrc
module load cuda/10.2
module load cudnn/7.6_cuda-10.2

#! Work directory (i.e. where the job will run):
workdir="$SLURM_SUBMIT_DIR"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory
                             # in which sbatch is run.
if [[ ! "$workdir" ]]; then
    workdir="$(pwd)"
fi

cd $workdir
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

# Test variables
echo -e "\nPrinting some test variables\n"
echo "PATH"
echo "$PATH"
echo "nvidia-smi"
nvidia-smi



######################################
### Joosts script starts from here ###
######################################

set -e # fail fully on first line failure

if [ -z "$SLURM_ARRAY_TASK_ID" ]
then
    # Not in Slurm Job Array - running in single mode
    JOB_ID=$SLURM_JOB_ID

    # Just read in what was passed over cmdline
    JOB_CMD="${@}"
else
    # In array
    JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

    # Get the line corresponding to the task id
    JOB_CMD=$(head -n ${SLURM_ARRAY_TASK_ID} "$1" | tail -1)
fi

# Submit the job
CMD="python $JOB_CMD"
echo $CMD
$CMD