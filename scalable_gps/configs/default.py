import ml_collections
from uci_datasets import all_datasets
import jax.random as jr


def get_dataset_config(name):
    config = ml_collections.ConfigDict()
    if name == "toy_sin":
        config.n = 100
        config.noise_scale = 1.0
        config.n_test = 500
        config.n_per_period = 100
        config.key = 42
    elif name in all_datasets.keys():
        pass

    return config


def get_config():
    config = ml_collections.ConfigDict()

    # Saving configs
    config.save_dir = "results/toy_sin"

    config.seed = 12345

    config.compute_exact_soln = True
    config.use_tpu = True

    # Data Configs
    config.dataset_name = "toy_sin"

    config.dataset_config = ml_collections.ConfigDict()
    config.dataset_config = get_dataset_config(config.dataset_name)

    config.dataset_config.normalise = True

    # Kernel Configs
    config.kernel_config = ml_collections.ConfigDict()
    config.kernel_config.signal_scale = 1.0
    config.kernel_config.length_scale = 1.0

    # Copy kernel configs to feature configs
    config.feature_config = config.kernel_config.copy_and_resolve_references()

    config.train_config = ml_collections.ConfigDict()

    # Full-batch training configs that get passed
    config.train_config.learning_rate = 1e-2
    config.train_config.momentum = 0.9
    config.train_config.polyak = 1e-3
    config.train_config.iterations = 50000
    config.train_config.batch_size = 4
    config.train_config.eval_every = 100
    # RFF Configs
    config.train_config.num_features = 100
    config.train_config.recompute_features = True

    config.optimiser = "sgd"
    config.sampling_loss_objective = 1

    # Wandb Configs
    config.wandb = ml_collections.ConfigDict()
    config.wandb.log = False
    config.wandb.project = "scalable-gps"
    config.wandb.entity = "shreyaspadhy"
    config.wandb.code_dir = "/home/shreyaspadhy_gmail_com/scalable-gaussian-processes"

    return config
