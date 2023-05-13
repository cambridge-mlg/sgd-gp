import subprocess

# methods = ['sgd', 'cg', 'precondcg', 'vi']
# methods = ['sgd', 'cg']
methods = ["sgd"]
splits = list(range(5))

datasets = ['pol', 'elevators', 'bike', 'protein', 'keggdirected'] 
datasets = datasets[::-1]

# , 'kin40k'] # 
# datasets = ['houseelectric']

low_noise = True

for dataset in datasets:
    for method in methods:
        for split in splits:
            name = f"{dataset}_{method}_{split}"
            command = f"python scripts/obtain_samples.py 9835d6db89010f73306f92bb9a080c9751b25d28 --config configs/default.py:{dataset} --config.model_name {method} --config.wandb.log --config.dataset_config.split {split} --noconfig.compute_exact_soln --config.wandb.log_artifact"
            
            if dataset in ['keggdirected', 'elevators']:
                command += ' --config.sampling_config.learning_rate 0.001'
            else:
                command += ' --config.sampling_config.learning_rate 0.1'

            name = f'samples_final_{name}'
            if low_noise:
                name += '_low_noise'
                command += " --config.override_noise_scale 0.001"

            command += f" --config.wandb.name {name}"
            subprocess.run(command.split())