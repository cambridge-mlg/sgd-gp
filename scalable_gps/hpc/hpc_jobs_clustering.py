from pathlib import Path

EXPERIMENT_NAME = "clustering"
JOBS_FOLDER = f"hpc-jobs-clustering"
DELETE_PREV_FOLDER = True
SCRIPT = "/home/ja666/rds/hpc-work/scalable-gaussian-processes/scalable_gps/scripts/dataset_clustering.py"
TIME = "03:00:00"

datasets = [
    "pol",
    "elevators",
    "bike",
    "kin40k",
    "protein",
    "keggdirected",
    # "slice",
    # "keggundirected",
    "3droad",
    "song",
    "buzz",
    "houseelectric",
]

LENGHTSCALE_RATIOS = [1, 2, 4]

n_splits = 10

# delete and create job folder
jobsfolder = Path(f"./{JOBS_FOLDER}")
if jobsfolder.exists() and DELETE_PREV_FOLDER:
    for jobfile in jobsfolder.glob("*"):
        jobfile.unlink()
    jobsfolder.rmdir()
jobsfolder.mkdir(exist_ok=True)

# create job file
jobsfile = jobsfolder / f"{EXPERIMENT_NAME}_{TIME}.txt"
if jobsfile.exists():
    jobsfile.unlink()
jobsfile.touch()

# write job commands
with open(jobsfile, "w") as f:
    for dataset in datasets:
        for split in range(n_splits):
            for ratio in LENGHTSCALE_RATIOS:
                # pass wandb API key as argv[0]
                line = (
                    f"{SCRIPT} 199c473d24b2682c6b0291b49241f5781e65a655 "
                    f"--config /home/ja666/rds/hpc-work/scalable-gaussian-processes/scalable_gps/configs/clustering_config.py "
                    f"--config.dataset_name {dataset} "
                    f"--config.dataset_split {split} "
                    f"--config.lengthscale_ratio {ratio} "
                    f"--config.wandb.log "
                    f"--config.wandb.name clustering_{dataset}_split={split}_ratio={ratio}\n"
                )
                f.write(line)
