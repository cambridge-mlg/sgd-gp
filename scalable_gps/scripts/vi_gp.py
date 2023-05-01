
import jax
import jax.numpy as jnp
import jax.random as jr
import ml_collections.config_flags
import wandb
from absl import app, flags

from scalable_gps import kernels
from scalable_gps.data import get_dataset
from scalable_gps.eval_utils import RMSE
from scalable_gps.models.vi_gp_model import SVGPModel
from scalable_gps.utils import HparamsTuple, flatten_nested_dict, get_tuned_hparams, setup_training, update_config_dict

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

        print(f"train_ds.x.shape: {train_ds.x.shape}")
        print(f"train_ds.y.shape: {train_ds.y.shape}")

        try:
            hparams = get_tuned_hparams(config.dataset_name, config.dataset_config.split)
        except wandb.CommError:
            print("Could not fetch hparams from wandb. Using default values.")
        
            hparams = HparamsTuple(
                length_scale=jnp.array(config.kernel_config.length_scale),
                signal_scale=config.kernel_config.signal_scale,
                noise_scale=config.dataset_config.noise_scale,)
        
        print(hparams)
        
        train_key, sampling_key = jr.split(jr.PRNGKey(config.seed))
        
        kernel_init_fn = getattr(kernels, config.kernel_name)
        
        kernel_config = {'signal_scale': hparams.signal_scale, 'length_scale': hparams.length_scale}
        kernel = kernel_init_fn(kernel_config)
        model = SVGPModel(hparams.noise_scale, kernel, config=config, kernel_config=kernel_config)

        model.compute_representer_weights(train_key, train_ds, test_ds, config.vi_config)
        
        y_pred = model.predictive_mean(train_ds, test_ds, recompute=False)
        test_rmse = RMSE(test_ds.y, y_pred, mu=train_ds.mu_y, sigma=train_ds.sigma_y)
        normalised_test_rmse = RMSE(test_ds.y, y_pred)

        zero_mean_posterior_samples = model.compute_posterior_samples(sampling_key, config.sampling_config.n_samples)
        
        y_pred_var = model.predictive_variance(train_ds, test_ds, recompute=False)
        
        y_pred_var_samples = model.predictive_variance_samples(zero_mean_posterior_samples)
        
        print(f'y_pred_var.shape: {y_pred_var.shape}')
        print(f'y_pred_var_samples.shape: {y_pred_var_samples.shape}')
        print(f"test_rmse_exact = {test_rmse}")
        wandb.log({"test_rmse": test_rmse,
                   "normalised_test_rmse": normalised_test_rmse})
        
        

if __name__ == "__main__":
    # Adds jax flags to the program.
    jax.config.config_with_absl()

    # Snippet to pass configs into the main function.
    def _main(argv):
        del argv
        config = FLAGS.config
        main(config)

    app.run(_main)