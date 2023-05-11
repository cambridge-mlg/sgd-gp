import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.use_tpu = False

    config.n_trees = 50
    config.n_neighbours = 100

    config.dataset_name = "kin40k"
    config.dataset_split = 0

    config.lengthscale_ratio = 2.0
    config.lengthscale_percentile = 25

    # WANDB
    config.wandb = ml_collections.ConfigDict()
    config.wandb.log = False
    config.wandb.project = "scalable-gps"
    config.wandb.entity = "shreyaspadhy"
    # TODO: change this to HPC dir
    config.wandb.code_dir = "/home/ja666/Code/scalable-gaussian-processes/"
    config.wandb.name = ""

    return config
