import jax
import jax.numpy as jnp
import jax.random as jr
import ml_collections.config_flags
import wandb
from absl import app, flags

from scalable_gps import kernels
from scalable_gps.data import get_dataset
from scalable_gps.eval_utils import RMSE
from scalable_gps.linear_model import marginal_likelihood
from scalable_gps.models.cg_gp_model import CGGPModel
from scalable_gps.models.exact_gp_model import ExactGPModel
from scalable_gps.utils import (
    ExactPredictionsTuple,
    HparamsTuple,
    flatten_nested_dict,
    get_tuned_hparams,
    setup_training,
    update_config_dict,
)

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
        
        kernel_init_fn = getattr(kernels, config.kernel_name)
        kernel = kernel_init_fn({'signal_scale': hparams.signal_scale, 'length_scale': hparams.length_scale})
        
        key = jr.PRNGKey(config.seed)
        _, sampling_key, key = jr.split(key, 3)

        # Compute exact solution
        exact_model, exact_metrics = None, None
        if config.compute_exact_soln:
            exact_model = ExactGPModel(hparams.noise_scale, kernel)

            exact_model.compute_representer_weights(train_ds)
            y_pred_exact = exact_model.predictive_mean(train_ds, test_ds)
            test_rmse_exact = RMSE(test_ds.y, y_pred_exact, mu=train_ds.mu_y, sigma=train_ds.sigma_y)
            normalised_test_rmse = RMSE(test_ds.y, y_pred_exact)

            mll = marginal_likelihood(train_ds.x, train_ds.y, exact_model.kernel.kernel_fn, hparams)
            print(f"test_rmse_exact = {test_rmse_exact}")
            wandb.log({"exact_gp/test_rmse": test_rmse_exact,
                    "exact_gp/normalised_test_rmse": normalised_test_rmse,
                    "exact_gp/mll": mll / train_ds.N})

            # Define exact metrics that we will use later to compare with stochastic solution
            exact_metrics = ExactPredictionsTuple(
                alpha=exact_model.alpha,
                y_pred_loc=y_pred_exact
            )
        
        cg_model = CGGPModel(hparams.noise_scale, kernel)

        metrics_list = ["test_rmse", "normalised_test_rmse"]
        if config.compute_exact_soln:
            metrics_list.extend(["alpha_diff", "y_pred_diff", "alpha_rkhs_diff"])

        cg_model.compute_representer_weights(
            train_ds, 
            test_ds, 
            config.cg_config, 
            metrics_list, 
            metrics_prefix="train", 
            exact_metrics=exact_metrics if config.compute_exact_soln else None)

        y_pred = cg_model.predictive_mean(train_ds, test_ds)
        test_rmse = RMSE(test_ds.y, y_pred, mu=train_ds.mu_y, sigma=train_ds.sigma_y)
        normalised_test_rmse = RMSE(test_ds.y, y_pred)

        mll = marginal_likelihood(train_ds.x, train_ds.y, cg_model.kernel.kernel_fn, hparams)
        print(f"test_rmse_cg = {test_rmse}")
        wandb.log({"cg_gp/test_rmse": test_rmse,
                   "cg_gp/normalised_test_rmse": normalised_test_rmse,
                   "cg_gp/mll": mll / train_ds.N})
        
        zero_mean_samples, alpha_samples = cg_model.compute_posterior_samples(
            sampling_key,
            n_samples=config.cg_sampling_config.n_samples,
            train_ds=train_ds,
            test_ds=test_ds,
            config=config.cg_sampling_config,
            use_rff=False,
            n_features=config.cg_sampling_config.n_features_prior_sample,
            zero_mean=True,
            metrics_list=metrics_list,
            metrics_prefix="sampling",
            compare_exact=True
        )

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