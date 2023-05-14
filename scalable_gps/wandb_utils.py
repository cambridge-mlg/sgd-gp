import wandb
from tqdm import tqdm


def load_runs_from_sweep(sweep_id, config_keys, metric_keys, entity = "shreyaspadhy", project="scalable-gps"):
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    runs = sweep.runs

    configs_and_metrics = []

    for run in tqdm(runs):
        # retrieve desired configs
        configs = {k: run.config[k] for k in config_keys}

        # retrieve desired metrics
        # using massive page_size makes this much faster (reduce if you have memory / bandwidth issues)
        history = run.scan_history(keys=metric_keys, page_size=100000000)
        # convert list of dicts into dict of lists
        metrics = {k: [d[k] for d in history] for k in metric_keys}
        
        configs_and_metrics.append((configs, metrics))

    return configs_and_metrics
