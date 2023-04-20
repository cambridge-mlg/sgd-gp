
import jax
from scalable_gps import kernels
import ml_collections.config_flags
import wandb
from absl import app, flags
from scalable_gps.data import get_dataset
from scalable_gps.eval_utils import RMSE
from scalable_gps.linear_model import marginal_likelihood
from scalable_gps.models import ExactGPModel
from scalable_gps.utils import flatten_nested_dict, get_tuned_hparams, setup_training, update_config_dict

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

        hparams = get_tuned_hparams(config.dataset_name, config.dataset_config.split)
        
        print(hparams)
        
        kernel_init_fn = getattr(kernels, config.kernel_name)
        
        kernel = kernel_init_fn({'signal_scale': hparams.signal_scale, 'length_scale': hparams.length_scale})
        exact_model = ExactGPModel(hparams.noise_scale, kernel)

        exact_model.compute_representer_weights(train_ds)
        y_pred = exact_model.predictive_mean(train_ds, test_ds)
        test_rmse = RMSE(test_ds.y, y_pred, mu=train_ds.mu_y, sigma=train_ds.sigma_y)
        normalised_test_rmse = RMSE(test_ds.y, y_pred)

        mll = marginal_likelihood(train_ds.x, train_ds.y, exact_model.kernel.kernel_fn, hparams)
        print(f"test_rmse_exact = {test_rmse}")
        wandb.log({"test_rmse": test_rmse,
                   "normalised_test_rmse": normalised_test_rmse,
                   "mll": mll / train_ds.N})

if __name__ == "__main__":
    # Adds jax flags to the program.
    jax.config.config_with_absl()

    # Snippet to pass configs into the main function.
    def _main(argv):
        del argv
        config = FLAGS.config
        main(config)

    app.run(_main)