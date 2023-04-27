import ml_collections

def get_config():

    config = ml_collections.ConfigDict()
    config.use_tpu = True

    config.seed = 1337
    config.D = 8
    config.kernel_name = "Matern32Kernel"
    config.signal_scale = 1.0
    config.length_scale = [0.1]
    config.noise_scale = 1e-3
    config.n_features = 1000
    config.n_init = 1000
    config.n_friends = 1000
    config.n_homies = 100
    config.n_besties = 1
    config.n_samples = 100
    config.find_friends_method = 'uniform'
    config.optim_lr = 1e-3
    config.optim_iters = 100
    config.iterations = 100

    config.wandb = ml_collections.ConfigDict()
    config.wandb.log = False
    config.wandb.project = "scalable-gps"
    config.wandb.entity = "shreyaspadhy"
    config.wandb.code_dir = "/home/shreyaspadhy_gmail_com/scalable-gaussian-processes"
    config.wandb.name = ""
    
    return config
