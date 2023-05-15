#!/bin/bash

# adapted from https://github.com/y0ast/slurm-for-ml/blob/master/run_file.sh

# Expects to be in the same folder as generic.sh

# Set number of GPUs and CPUs
gpus=1      # NOTE: batch size will be divided by number of GPUs
cpus=$((32 * $gpus))

# Check if provided directory exists
if [ ! -d "$1" ]
then
    echo "Error: directory passed does not exist"
    exit 1
fi

# Loop over files in the given directory
for FILE in $1*; do
    echo $FILE
    # This convoluted way of counting also works if a final EOL character is missing
    n_jobs=$(grep -c '^' "$FILE")

    # Set/determine number of jobs to run in parallel
    jobs_in_parallel=auto

    max_n_gpus=64
    if [ "$jobs_in_parallel" = "auto" ]; then
        max_n_jobs=$(($max_n_gpus / $gpus))
        # this computes min(max_n_jobs, n_jobs); see https://unix.stackexchange.com/a/186703
        jobs_in_parallel=$(( $max_n_jobs < $n_jobs ? $max_n_jobs : $n_jobs ))
    fi

    # Extract job time and name from filename
    # see https://stackoverflow.com/a/5257398 for an explanation of this code
    IFS='/'; arr=($FILE); unset IFS;
    FILE_=${arr[2]}
    IFS='.'; arr=($FILE_); unset IFS;
    FILE_=${arr[0]}
    IFS='_'; arr=($FILE_); unset IFS;
    NAME=${arr[0]}
    TIME=${arr[1]}
    echo $NAME
    echo $arr
    # Define and create logging directory
    log_dir="logs/$NAME"
    mkdir -p $log_dir
    log_file="$log_dir/%x_%A_%a.out"
    err_file="$log_dir/%x_%A_%a.err"

    cmd="sbatch --array=1-${n_jobs}%${jobs_in_parallel} --time $TIME --job-name $NAME --gres=gpu:$gpus --cpus-per-task=$cpus -o $log_file -e $err_file $(dirname "$0")/generic.sh "$FILE""
    echo $cmd
    eval $cmd
done
