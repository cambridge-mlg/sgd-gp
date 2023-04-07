from pathlib import Path

EXPERIMENT_NAME = "mll-optim"
JOBS_FOLDER = f'hpc-jobs-mll-optim'
DELETE_PREV_FOLDER = True
SCRIPT = "mll_optim.py"
TIME = "10:00:00"

datasets = ["pol",
            "elevators",
            "bike",
            "kin40k",
            "protein",
            "keggdirected",
            # "slice",
            "keggundirected",
            "3droad",
            "song",
            "buzz",
            "houseelectric"]

n_splits = 10
n_subsample = 10

# delete and create job folder
jobsfolder = Path(f'./{JOBS_FOLDER}')
if jobsfolder.exists() and DELETE_PREV_FOLDER:
    for jobfile in jobsfolder.glob('*'):
        jobfile.unlink()
    jobsfolder.rmdir()
jobsfolder.mkdir(exist_ok=True)

# create job file
jobsfile = jobsfolder / f'{EXPERIMENT_NAME}_{TIME}.txt'
if jobsfile.exists():
    jobsfile.unlink()
jobsfile.touch()

# write job commands
with open(jobsfile, 'w') as f:
    for dataset in datasets:
        for split in range(n_splits):
            for seed in range(n_subsample):
                # pass wandb API key as argv[0]
                line = (f'{SCRIPT} 17bb59c7a4e186597fe98347175642c99c8403e2 '
                        f'--config configs/default.py:{dataset} '
                        f'--config.dataset_config.split {split} '
                        f'--config.mll_config.subsample_seed {seed} '
                        f'--config.wandb.log '
                        f'--config.wandb.code_dir /home/jal232/Code/scalable-gaussian-processes '
                        f'--config.wandb.name hparams_{dataset}_split={split}_seed={seed}\n')
                f.write(line)
