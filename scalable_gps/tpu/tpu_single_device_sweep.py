import subprocess

methods = ['sgd', 'cg', 'precondcg', 'vi']
splits = list(range(10))
datasets = ['pol', 'elevators', 'bike']
for dataset in datasets:
    for method in methods:
        for split in splits:
            name = f"{dataset}_{method}_{split}"
            command = f"python scripts/obtain_mean.py 9835d6db89010f73306f92bb9a080c9751b25d28 --config configs/default.py:{dataset} --config.model_name {method} --config.wandb.log --config.dataset_config.split {split} --config.wandb.name {name}"
            subprocess.run(command.split())