from pathlib import Path

import jax
import jax.random as jr
import ml_collections.config_flags
from absl import app, flags

import wandb
from data import get_dataset
from kernels import RBF, RFF
from models import ExactGPModel, SamplingGPModel
from utils import flatten_nested_dict, update_config_dict, setup_training

ml_collections.config_flags.DEFINE_config_file(
    "config",
    "configs/default.py",
    "Training configuration.",
    lock_config=True,
)

FLAGS = flags.FLAGS


def main(config):
    wandb_kwargs = {
        "project": config.wandb.project,
        "entity": config.wandb.entity,
        "config": flatten_nested_dict(config.to_dict()),
        "mode": "online" if config.wandb.log else "disabled",
        "settings": wandb.Settings(code_dir=config.wandb.code_dir),
    }
    with wandb.init(**wandb_kwargs) as run:
        setup_training(run)
        # If there are any config values dependent on sweep values, recompute them here.
        computed_configs = {}
        update_config_dict(config, run, computed_configs)

        print(config)
        train_ds, test_ds = get_dataset(config.dataset_name, **config.dataset_config)
        
        print(f'train_ds.x.shape: {train_ds.x.shape}')
        print(f'train_ds.y.shape: {train_ds.y.shape}')

        def kernel_fn(x, y):
            return RBF(x, y, s=config.signal_scale, l=config.length_scale)
        def feature_fn(key, n_features, x):
            return RFF(key, n_features, x, s=config.signal_scale, l=config.length_scale)

        save_dir = Path(config.save_dir).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Compute exact solution
        exact_model = None
        if config.compute_exact_soln is True:
            exact_model = ExactGPModel(config.dataset_config.noise_scale, kernel_fn)
            exact_model.compute_representer_weights(train_ds)
            
            test_rmse_exact, y_pred_exact = exact_model.calculate_test_rmse(train_ds, test_ds)
            print(f'test_rmse_exact = {test_rmse_exact}')
            wandb.log({"test_rmse_exact": test_rmse_exact})
            compare_exact_vals = [exact_model.alpha, y_pred_exact, test_rmse_exact]
        # Compute stochastic optimised solution
        key = jr.PRNGKey(config.seed)
        model = SamplingGPModel(config.dataset_config.noise_scale, kernel_fn, feature_fn)
        
        model.compute_representer_weights(
            train_ds, test_ds, config.train_config, key, 
            compare_exact_vals=compare_exact_vals if config.compute_exact_soln else None)
        
        
        # Compute a posterior sample
        post_sample = model.compute_posterior_sample(train_ds, config.train_config, 1, key)
        
        return post_sample
        
        
        
    

if __name__ == "__main__":
    # Adds jax flags to the program.
    jax.config.config_with_absl()

    # Snippet to pass configs into the main function.
    def _main(argv):
        del argv
        config = FLAGS.config
        main(config)

    app.run(_main)