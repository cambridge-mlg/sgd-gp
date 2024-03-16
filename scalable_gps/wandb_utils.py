import wandb
from tqdm import tqdm


def _load(runs, config_keys, metric_keys, page_size=100000000):
    configs_and_metrics = []

    for run in tqdm(runs):
        if run.state != "finished":
            print(f"Skipping run '{run.id}' because it '{run.state}'.")
            continue
        # retrieve desired configs
        configs = {k: run.config[k] for k in config_keys}

        # retrieve desired metrics
        # using massive page_size makes this much faster (reduce if you have memory / bandwidth issues)
        history = run.scan_history(keys=metric_keys, page_size=page_size)
        # convert list of dicts into dict of lists
        # metrics = dict()
        # for k in metric_keys:
        #     metrics[k] = []
        #     for d in history:
        #         if k in d.keys():
        #             metrics[k].append(d[k])

        metrics = {k: [d[k] for d in history] for k in metric_keys}

        configs_and_metrics.append((configs, metrics))

    return configs_and_metrics


def load_runs_from_sweep(
    sweep_id, config_keys, metric_keys, entity="shreyaspadhy", project="scalable-gps"
):
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    runs = sweep.runs
    return _load(runs, config_keys, metric_keys)


def load_runs_from_regex(
    regex, config_keys, metric_keys, entity="shreyaspadhy", project="scalable-gps"
):
    api = wandb.Api()
    runs = api.runs(
        path=f"{entity}/{project}", filters={"display_name": {"$regex": regex}}
    )
    return _load(runs, config_keys, metric_keys)
