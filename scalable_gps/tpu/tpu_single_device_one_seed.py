import subprocess

methods = ['sgd', 'cg']
splits = list(range(1))
datasets = ['kin40k', 'protein', 'keggdirected']
low_noise = True

for dataset in datasets:
    for method in methods:
        for split in splits:
            name = f"{dataset}_{method}_{split}"
            command = f"python scripts/obtain_mean.py 9835d6db89010f73306f92bb9a080c9751b25d28 --config configs/default.py:{dataset} --config.model_name {method} --config.wandb.log --config.dataset_config.split {split} --noconfig.compute_exact_soln"
            
            name = f'debug_sweep_{name}'
            if low_noise:
                name += '_low_noise'
                command += " --config.override_noise_scale 0.001"

            command += f" --config.wandb.name {name}"
            
            subprocess.run(command.split())