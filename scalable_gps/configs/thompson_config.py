import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.use_tpu = False
    config.compute_exact_soln = False

    config.thompson = ml_collections.ConfigDict()
    config.thompson.use_tpu = False

    config.thompson.seed = 0
    config.thompson.D = 8 # try 8, 16

    config.thompson.model_name = "SGDGP"
    config.thompson.kernel_name = "Matern32Kernel" # try 5/2
    config.thompson.signal_scale = 1.0
    config.thompson.length_scale = 0.5 # try 0.1, 0.2, 0.3, 0.4, 0.5

    config.thompson.noise_scale = 1e-3

    config.thompson.iterations = 30

    config.thompson.n_features = 5000

    config.thompson.n_init = 50000 # 50k
    config.thompson.init_method = "uniform" # try "trunc_normal"

    config.thompson.friends_iterations = 30
    config.thompson.n_friends = 50000
    config.thompson.n_homies = 30 // config.thompson.friends_iterations
    config.thompson.n_besties = 1

    config.thompson.n_samples = 1000

    config.thompson.find_friends_method = "nearby" # uniform

    config.thompson.optim_lr = 1e-3 # try cosine annealing
    config.thompson.optim_iters = 100

    config.thompson.grid_search = config.thompson.D == 2
    config.thompson.grid_search_dim = 400

    config.thompson.minval = 0.0
    config.thompson.maxval = 1.0

    ####### CG #

    config.cg_config = ml_collections.ConfigDict()

    config.cg_config.batch_size = 1
    config.cg_config.maxiter = 10 # 10, 50
    config.cg_config.eval_every = 1

    config.cg_config.tol = 0.0
    config.cg_config.atol = 0.0

    config.cg_config.preconditioner = False
    config.cg_config.pivoted_chol_rank = 100
    config.cg_config.pivoted_diag_rtol = 1e-3
    config.cg_config.pivoted_jitter = 1

    config.cg_config.loss_objective = 2

    ### SGD #

    config.train_config = ml_collections.ConfigDict()

    # RFF Configs
    config.train_config.n_features_optim = 100
    config.train_config.recompute_features = True

    # Optimisation Configs

    config.train_config.iterations = 1000 # 10k, 50k
    config.train_config.batch_size = 500
    config.train_config.eval_every = 100
    config.train_config.time_budget_in_seconds = 0.
    config.train_config.eval_every_in_seconds = 0.

    config.train_config.iterative_idx = True
    config.train_config.learning_rate = 5e-1
    config.train_config.momentum = 0.9
    config.train_config.nesterov = True
    config.train_config.polyak = 100 / config.train_config.iterations
    config.train_config.absolute_clipping = 0.1 # -1 to avoid clipping
    config.train_config.lr_schedule_name = None # "linear_schedule"
    config.train_config.lr_schedule_config = ml_collections.ConfigDict()

    # Sampling
    config.sampling_config = ml_collections.ConfigDict()
    config.sampling_config = config.train_config.copy_and_resolve_references()

    config.sampling_config.iterative_idx = True
    config.sampling_config.learning_rate = 1e-3 # double learning rate
    config.sampling_config.loss_objective = 2

    # VI

    config.vi_config = ml_collections.ConfigDict()
    config.vi_config = config.thompson.copy_and_resolve_references()

    config.vi_config.iterations = 50000 # small compute: 20k, large compute 100x
    config.vi_config.batch_size = 500
    config.vi_config.num_inducing_points = 1024
    config.vi_config.inducing_init = "kmeans"
    config.vi_config.learning_rate = 5e-4
    config.vi_config.absolute_clipping = 0.1
    config.kernel_name = config.thompson.kernel_name

    # WANDB
    config.wandb = ml_collections.ConfigDict()
    config.wandb.log = True
    config.wandb.project = "scalable-gps"
    config.wandb.entity = "shreyaspadhy"
    # TODO: change this to HPC dir
    config.wandb.code_dir = "/home/jal232/Code/scalable-gaussian-processes/"
    config.wandb.name = ""

    return config
