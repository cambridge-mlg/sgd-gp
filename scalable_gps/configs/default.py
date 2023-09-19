import ml_collections
from uci_datasets import all_datasets
import jax.numpy as jnp

PROTEIN_DATASET_HPARAMS = {
    'tanimoto_ESR2': {'mean': -6.73,
                      'noise_scale': jnp.sqrt(0.261),
                       'signal_scale': 0.706},
    'tanimoto_F2': {'mean': -6.14,
                    'noise_scale': jnp.sqrt(0.0899),
                    'signal_scale': 0.356},
    'tanimoto_KIT': {'mean': -6.39,
                    'noise_scale': jnp.sqrt(0.112),
                    'signal_scale': 0.679},
    'tanimoto_PARP1': {'mean': -6.95,
                    'noise_scale': jnp.sqrt(0.0238),
                    'signal_scale': 0.56},
    'tanimoto_PGR': {'mean': -7.08,
                    'noise_scale': jnp.sqrt(0.332),
                    'signal_scale': 0.63},
}

def get_dataset_config(name):
    config = ml_collections.ConfigDict()
    if name == "toy_sin":
        config.n = 100
        config.noise_scale = 1.0
        config.n_test = 500
        config.n_per_period = 100
        config.seed = 42
        config.input_dim = 1
    elif name in all_datasets.keys():
        config.input_dim = all_datasets[name][-1]
        config.noise_scale = 1.0
    elif 'tanimoto' in name:
        config.input_dim = 1024
        config.noise_scale = 0.5
        config.dataset_dir = "./datafiles"
        config.n_train = None
        if name in PROTEIN_DATASET_HPARAMS.keys():
            config.data_target_mean = PROTEIN_DATASET_HPARAMS[name]['mean']
        else:
            config.data_target_mean = None
        config.binarize = False
    return config


def get_config(config_string):
    config = ml_collections.ConfigDict()

    d_name = config_string.split(".")[0]

    config.override_d_name = ""
    config.override_d_binarize = False
    
    # Saving configs
    config.save_dir = f"results/{d_name}"

    config.seed = 12345

    config.compute_exact_soln = True
    config.use_tpu = True
    config.override_noise_scale = -1.0

    config.model_name = "sgd"
    # Data Configs
    config.dataset_name = d_name

    config.dataset_config = ml_collections.ConfigDict()
    config.dataset_config = get_dataset_config(config.dataset_name)

    config.dataset_config.split = 0
    config.dataset_config.normalise = True

    # Kernel Configs
    config.kernel_name = "Matern32Kernel"
    config.kernel_config = ml_collections.ConfigDict()
    config.kernel_config.use_ard = True
    config.kernel_config.signal_scale = 1.0

    length_scale_dim = (
        config.dataset_config.input_dim if config.kernel_config.use_ard else 1
    )
    config.kernel_config.length_scale = length_scale_dim * [1.0]

    config.vi_config = ml_collections.ConfigDict()
    config.vi_config.iterations = 10000
    config.vi_config.batch_size = 100
    config.vi_config.num_inducing_points = 1024
    config.vi_config.inducing_init = "kmeans"
    config.vi_config.annoy_pre_clustering = False
    config.vi_config.clustering_length_scale_ratio = 1.0
    config.vi_config.learning_rate = 1e-3
    config.vi_config.absolute_clipping = 0.1
    config.vi_config.use_exact_pred_variance = False
    config.train_config = ml_collections.ConfigDict()

    # Full-batch training configs that get passed
    config.train_config.iterations = 100000
    config.train_config.batch_size = 512
    config.train_config.eval_every = 1000
    config.train_config.preempt_safe = False
    config.train_config.time_budget_in_seconds = 0.0
    config.train_config.eval_every_in_seconds = 0.0
    # RFF Configs
    config.train_config.n_features_optim = 100
    config.train_config.recompute_features = True

    config.train_config.rff_modulo_value = 8

    # Optimisation Configs
    config.train_config.iterative_idx = True
    config.train_config.learning_rate = 5e-1
    config.train_config.momentum = 0.9
    config.train_config.nesterov = True
    config.train_config.grad_variant = 'vanilla' # 'vanilla', 'batch_kvp', 'batch_err', 'batch_all', 'random_kvp'
    # TODO: Calculate polyak dynamically.
    config.train_config.polyak = 100 / config.train_config.iterations
    config.train_config.absolute_clipping = 0.1  # -1 to avoid clipping

    config.train_config.lr_schedule_name = None  # "linear_schedule"
    config.train_config.lr_schedule_config = ml_collections.ConfigDict()

    if config.train_config.lr_schedule_name == "linear_schedule":
        config.train_config.lr_schedule_config.decay_rate = 1 / 33
        config.train_config.lr_schedule_config.transition_steps = int(
            config.train_config.iterations * 0.95
        )  # I set this to N steps * 0.75
        config.train_config.lr_schedule_config.end_value = (
            config.train_config.learning_rate / 33
        )

    config.sampling_config = config.train_config.copy_and_resolve_references()
    config.sampling_config.n_samples = 64
    # Full-batch training configs that get passed
    config.sampling_config.iterative_idx = True
    config.sampling_config.learning_rate = 1e-1
    config.sampling_config.momentum = 0.9
    config.sampling_config.nesterov = True
    config.sampling_config.grad_variant = 'vanilla' # 'vanilla', 'batch_kvp', 'batch_err', 'batch_all', 'random_kvp'
    config.sampling_config.polyak = 100 / config.sampling_config.iterations
    config.sampling_config.iterations = 100000
    config.sampling_config.batch_size = 512
    config.sampling_config.eval_every = 1000
    # RFF Configs
    config.sampling_config.n_features_prior_sample = 2000
    config.sampling_config.n_features_optim = 100
    config.sampling_config.recompute_features = True
    config.sampling_config.loss_objective = 2
    config.sampling_config.use_cholesky_prior_sample = False

    config.sampling_config.absolute_clipping = 0.1  # -1 to avoid clipping

    config.mll_config = ml_collections.ConfigDict()
    config.mll_config.learning_rate = 0.1
    config.mll_config.iterations = 300
    config.mll_config.eval_every = 1

    config.mll_config.init_length_scale = length_scale_dim * [0.0]

    config.mll_config.init_signal_scale = 0.0
    config.mll_config.init_noise_scale = 0.0

    config.mll_config.subsample_seed = 0
    config.mll_config.n_subsample = 10000

    config.cg_config = ml_collections.ConfigDict()
    config.cg_config.batch_size = 1
    config.cg_config.tol = 1e-2
    config.cg_config.maxiter = 1000
    config.cg_config.atol = 0.0
    config.cg_config.eval_every = 1
    config.cg_config.preconditioner = False
    config.cg_config.pivoted_chol_rank = 100
    config.cg_config.pivoted_diag_rtol = 1e-3
    config.cg_config.pivoted_jitter = 1.0
    config.cg_config.loss_objective = 2
    config.cg_config.preempt_safe = False

    config.cg_sampling_config = ml_collections.ConfigDict()
    config.cg_sampling_config.batch_size = 1
    config.cg_sampling_config.n_features_prior_sample = 2000
    config.cg_sampling_config.n_samples = 64
    config.cg_sampling_config.tol = 1e-3
    config.cg_sampling_config.maxiter = 1000
    config.cg_sampling_config.atol = 0.0
    config.cg_sampling_config.eval_every = 1
    config.cg_sampling_config.preconditioner = True
    config.cg_sampling_config.pivoted_chol_rank = 100
    config.cg_sampling_config.pivoted_diag_rtol = 1e-3
    config.cg_sampling_config.pivoted_jitter = 1.0
    config.cg_sampling_config.loss_objective = 2

    # Wandb Configs
    config.wandb = ml_collections.ConfigDict()
    config.wandb.log = True
    config.wandb.log_artifact = True
    config.wandb.project = "faster-sgd-gp"
    config.wandb.entity = "jandylin"
    config.wandb.code_dir = "/home/jal232/Code/scalable-gaussian-processes" # TODO: use os utility to get this
    config.wandb.name = ""

    return config
