from pathlib import Path

EXPERIMENT_NAME = "thompson-hparams"
JOBS_FOLDER = 'hpc-jobs-thompson-hparams'
DELETE_PREV_FOLDER = True
SCRIPT = "scripts/thompson_sampling.py"
TIME = "24:00:00"

model_names = ["CGGP", "SGDGP", "SVGP"]
dims = [8, 16, 32, 64]
length_scales = [0.25, 0.5, 0.75, 1.0, 1.25]

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
    for model_name in model_names:
        for D in dims:
            for length_scale in length_scales:
                # pass wandb API key as argv[1]
                line = (f"{SCRIPT} 17bb59c7a4e186597fe98347175642c99c8403e2 "
                        f"--config configs/thompson_config.py "
                        f"--config.thompson.model_name {model_name} "
                        f"--config.thompson.D {D} "
                        f"--config.thompson.length_scale '{length_scale},' "
                        f"--config.wandb.log "
                        f"--config.wandb.code_dir /home/jal232/Code/scalable-gaussian-processes "
                        f"--config.wandb.name TS_hparams_model={model_name}_D={D}_ls={length_scale}\n")
                f.write(line)
