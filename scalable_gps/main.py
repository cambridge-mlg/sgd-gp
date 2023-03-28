from pathlib import Path

import jax
import jax.random as jr
import ml_collections.config_flags
from absl import app, flags
from data import get_dataset
from kernels import RBFKernel
from models import ExactGPModel, SGDGPModel
from utils import ExactValsTuple, flatten_nested_dict, setup_training, update_config_dict

import wandb

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

        kernel = RBFKernel(config.kernel_config)

        save_dir = Path(config.save_dir).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)

        key = jr.PRNGKey(config.seed)
        optim_key, sampling_key, key = jr.split(key, 3)

        # Compute exact solution
        exact_model, compare_exact_vals = None, None
        if config.compute_exact_soln is True:
            exact_model = ExactGPModel(config.dataset_config.noise_scale, kernel)
            exact_model.compute_representer_weights(train_ds)

            test_rmse_exact, y_pred_exact = exact_model.calculate_test_rmse(
                train_ds, test_ds)
            
            zero_mean_posterior_samples, alpha_samples_exact = exact_model.compute_zero_mean_samples(
                sampling_key, 
                n_samples=10,
                train_ds=train_ds,
                test_ds=test_ds,
                n_features=config.train_config.num_features,
                use_rff=False
            )
            print(f"zero_mean_posterior_samples.shape: {zero_mean_posterior_samples.shape}")
            print(f"alpha_samples_exact.shape: {alpha_samples_exact.shape}")
            
            print(f"test_rmse_exact = {test_rmse_exact}")
            wandb.log({"test_rmse_exact": test_rmse_exact})
            compare_exact_vals = ExactValsTuple(
                alpha=exact_model.alpha,
                y_pred=y_pred_exact,
                test_rmse=test_rmse_exact,
                alpha_sample=alpha_samples_exact,
                y_pred_sample=zero_mean_posterior_samples,
            )

        # Compute stochastic optimised solution
        model = SGDGPModel(config.dataset_config.noise_scale, kernel)

        metrics = ["loss", "grad_var", "test_rmse"]
        if config.compute_exact_soln:
            metrics.extend(["alpha_diff", "y_pred_diff", "test_rmse_diff"])

        model.compute_representer_weights(
            optim_key,
            train_ds,
            test_ds,
            config.train_config,
            metrics=metrics,
            metrics_prefix="train",
            compare_exact_vals=compare_exact_vals if config.compute_exact_soln else None,
        )

        # TODO: vmap and pmap sampling to obtain multiple samples in parallel
        sampling_metrics = ["loss", "grad_var", "test_rmse"]
        if config.compute_exact_soln:
            sampling_metrics.extend(
                ["alpha_sample_diff", "y_pred_diff", "loss_diff", "test_rmse_diff"]
            )
        
        zero_mean_samples, alpha_samples = model.compute_zero_mean_samples(
            sampling_key, 
            n_samples=10,
            train_ds=train_ds,
            test_ds=test_ds,
            config=config.sampling_config,
            sampling_metrics=sampling_metrics,
            use_rff=False,
            compare_exact_vals=compare_exact_vals if config.compute_exact_soln else None,
        )

        print(f"zero_mean_samples.shape: {zero_mean_samples.shape}")
        print(f"alpha_samples.shape: {alpha_samples.shape}")

        

        return zero_mean_samples, alpha_samples


if __name__ == "__main__":
    # Adds jax flags to the program.
    jax.config.config_with_absl()

    # Snippet to pass configs into the main function.
    def _main(argv):
        del argv
        config = FLAGS.config
        main(config)

    app.run(_main)
