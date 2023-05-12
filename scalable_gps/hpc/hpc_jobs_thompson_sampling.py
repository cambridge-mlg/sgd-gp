from pathlib import Path

EXPERIMENT_NAME = "thompson-sampling"
JOBS_FOLDER = 'hpc-jobs-thompson-sampling'
DELETE_PREV_FOLDER = True
SCRIPT = "scripts/thompson_sampling.py"
TIME = "24:00:00"

initial_seed = 0
n_seeds = 10

# run = 0: CG maxiter 20, SGD 10k iters, SVGP 10k iters
# run = 1: CG maxiter 100, SGD 100k iters, SVGP 100k iters
# run = 2: SVGP 50k iters, 5e-4
run = 2
model_names = ["RandomSearch"]
dims = [8,]
length_scales = [0.1, 0.2, 0.3, 0.4, 0.5]

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
    for seed in range(n_seeds):
        for model_name in model_names:
            for D in dims:
                for length_scale in length_scales:
                    # pass wandb API key as argv[1]
                    line = (f"python {SCRIPT} 17bb59c7a4e186597fe98347175642c99c8403e2 "
                            f"--config configs/thompson_config.py "
                            f"--config.thompson.seed {initial_seed + seed} "
                            f"--config.thompson.model_name {model_name} "
                            f"--config.thompson.D {D} "
                            f"--config.thompson.length_scale '{length_scale},' "
                            f"--config.wandb.log "
                            f"--config.wandb.code_dir /home/jal232/Code/scalable-gaussian-processes "
                            f"--config.wandb.name TS_run={run}_model={model_name}_D={D}_ls={length_scale}_{seed}\n")
                    f.write(line)
