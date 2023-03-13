from pathlib import Path

import jax
import ml_collections.config_flags
from absl import app, flags

import wandb
from data import get_dataset
from kernels import RBF, RFF
from train_utils import compute_exact_solution, compute_optimised_solution
from utils import flatten_nested_dict, update_config_dict

ml_collections.config_flags.DEFINE_config_file(
    "config",
    "configs/toy_sin.py",
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
        alpha_exact = None
        if config.compute_exact_soln is True:
            alpha_exact, y_pred_exact, test_rmse_exact = compute_exact_solution(
                train_ds, test_ds, kernel_fn, noise_scale=config.noise_scale)

            wandb.log({"test_rmse_exact": test_rmse_exact})
        
        # Compute stochastic optimised solution
        alpha_polyak, aux = compute_optimised_solution(
            config, train_ds, test_ds, K, kernel_fn, feature_fn, alpha_exact)
        
        
    

if __name__ == "__main__":
    # Adds jax flags to the program.
    jax.config.config_with_absl()

    # Snippet to pass configs into the main function.
    def _main(argv):
        del argv
        config = FLAGS.config
        main(config)

    app.run(_main)