from pathlib import Path

low_noise = False

EXPERIMENT_NAME = "obtain-samples-houseelectric"
JOBS_FOLDER = f'hpc-jobs-obtain-samples-houseelectric'
DELETE_PREV_FOLDER = True
SCRIPT = "/home/sp2058/rds/rds-t2-cs117/sp2058/repos/scalable-gaussian-processes/scalable_gps/scripts/obtain_samples.py"
TIME = "18:00:00"

datasets = [
    # "pol",
    # "elevators",
    # "bike",
    # "kin40k",
    # "protein",
    # "keggdirected",
    # "slice",
    # "keggundirected",
    # "3droad",
    # "song",
    # "buzz",
    "houseelectric"
    ]

CG_BS = {
    "kin40k": 1,
    "3droad": 4096,
    "song": 512,
    "buzz": 512,
    "houseelectric": 512,}

CG_MAXITER = {
    "kin40k": 1000,
    '3droad': 1000,
    'song': 100,
    'buzz': 100,
    'houseelectric': 100,}

SGD_LR = {
    '3droad': 0.1,
    'song': 0.001,
    'buzz': 0.0001,
    'houseelectric': 0.001,}

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

methods = ['sgd']

# all_splits = [0, 1, 2, 3, 4]
all_splits = [0, 1, 2]


splits = {
    'cg': all_splits,
    'precondcg': all_splits,
    'sgd': all_splits,
    'vi': all_splits
    }


# write job commands
use_exact = [False]
with open(jobsfile, 'w') as f:
    for dataset in datasets:
        for method in methods:
             for split in splits[method]:
                for exact in use_exact:
                    # pass wandb API key as argv[0]
                    line = (f'python {SCRIPT} 9835d6db89010f73306f92bb9a080c9751b25d28 '
                            f'--config configs/default.py:{dataset} '
                            f'--config.model_name {method} '
                            f'--config.dataset_config.split {split} '
                            f'--config.wandb.log '
                            f'--config.wandb.code_dir /home/sp2058/rds/rds-t2-cs117/sp2058/repos/scalable-gaussian-processes '
                            # f'--config.wandb.log_artifact '
                            # f'--config.cg_sampling_config.batch_size {CG_BS[dataset]} '
                            # f'--config.cg_sampling_config.maxiter {CG_MAXITER[dataset]} '
                            f'--config.sampling_config.learning_rate {SGD_LR[dataset]} '
                            f'--noconfig.compute_exact_soln '
                            f'--config.sampling_config.eval_every 1000 ')
                    
                    name = f'samples_final_{dataset}_{method}_{split}'
                    if dataset == "houseelectric" and method == "vi":
                        line += '--config.vi_config.annoy_pre_clustering '
                    
                    if exact:
                        line += '--config.vi_config.use_exact_pred_variance '
                        name += '_exactpredvar'


                    line += f'--config.wandb.name {name}\n'
                    f.write(line)