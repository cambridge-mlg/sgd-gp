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
from scalable_gps.models.sgd_gp_model import SGDGPModel
from scalable_gps.models.vi_gp_model import SVGPModel
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

import pickle

FLAGS = flags.FLAGS


def main(config):
    wandb_kwargs = {
        "project": config.wandb.project,
        "entity": config.wandb.entity,
        "config": flatten_nested_dict(config.to_dict()),
        "name": config.wandb.name if config.wandb.name else None,
        "mode": "online" if config.wandb.log else "disabled",
        "settings": wandb.Settings(code_dir=config.wandb.code_dir),
    }
    with wandb.init(**wandb_kwargs) as run:
        setup_training(run)
        # If there are any config values dependent on sweep values, recompute them here.
        computed_configs = {}
        update_config_dict(config, run, computed_configs)

        print(config)
        
        # Obtain Dataset and HParams
        train_ds, test_ds = get_dataset(config.dataset_name, **config.dataset_config)

        try:
            hparams = get_tuned_hparams(config.dataset_name, config.dataset_config.split)
        except wandb.CommError:
            print("Could not fetch hparams from wandb. Using default values.")
        
            hparams = HparamsTuple(
                length_scale=jnp.array(config.kernel_config.length_scale),
                signal_scale=config.kernel_config.signal_scale,
                noise_scale=config.dataset_config.noise_scale,
                )
        if config.override_noise_scale > 0.:
            hparams = HparamsTuple(
                length_scale=hparams.length_scale,
                signal_scale=hparams.signal_scale,
                noise_scale=config.override_noise_scale)
        print(hparams)
        
        # Initialise Kernel
        kernel_init_fn = getattr(kernels, config.kernel_name)
        kernel = kernel_init_fn({'signal_scale': hparams.signal_scale, 'length_scale': hparams.length_scale})

        key = jr.PRNGKey(config.seed)
        optim_key, sampling_key, key = jr.split(key, 3)

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
            wandb.log({"exact/test_rmse": test_rmse_exact,
                    "exact/normalised_test_rmse": normalised_test_rmse,
                    "exact/mll": mll / train_ds.N})

            # Define exact metrics that we will use later to compare with stochastic solution
            exact_metrics = ExactPredictionsTuple(
                alpha=exact_model.alpha,
                y_pred_loc=y_pred_exact
            )

        # Compute stochastic optimised solution
        if config.model_name == "sgd":
            model = SGDGPModel(hparams.noise_scale, kernel)
            train_config = config.train_config
        elif config.model_name == "cg":
            model = CGGPModel(hparams.noise_scale, kernel)
            train_config = config.cg_config
            train_config.preconditioner = False
        elif config.model_name == "precondcg":
            model = CGGPModel(hparams.noise_scale, kernel)
            train_config = config.cg_config
            train_config.preconditioner = True
        elif config.model_name == "vi":
            train_config = config.vi_config
            kernel_config = {'signal_scale': hparams.signal_scale, 'length_scale': hparams.length_scale}
            model = SVGPModel(hparams.noise_scale, kernel, config, kernel_config)

        metrics_list = ["loss", "err", "reg", "normalised_test_rmse", "test_rmse"]
        if config.compute_exact_soln:
            metrics_list.extend(["alpha_diff", "y_pred_diff", "alpha_rkhs_diff"])

        # Compute the SGD MAP solution for representer weights.
        model.compute_representer_weights(
            optim_key,
            train_ds,
            test_ds,
            train_config,
            metrics_list=metrics_list,
            metrics_prefix="train",
            exact_metrics=exact_metrics if config.compute_exact_soln else None,
        )
        
        y_pred = model.predictive_mean(train_ds, test_ds)
        test_rmse = RMSE(test_ds.y, y_pred, mu=train_ds.mu_y, sigma=train_ds.sigma_y)
        normalised_test_rmse = RMSE(test_ds.y, y_pred)

        wandb.log({"test_rmse": test_rmse,
                   "normalised_test_rmse": normalised_test_rmse,})
        if config.wandb.log_artifact:
            # Use wandb artifacts to save model hparams for a given dataset split and subsample_idx.
            model_artifact = wandb.Artifact(
                f"model_{config.dataset_name}_{config.model_name}_{config.dataset_config.split}", type="model",
                description=f"Saved Model class for {config.dataset_name} dataset with method {config.model_name} on split {config.dataset_config.split}.",
                metadata={**{"dataset_name": config.dataset_name, "model_name": config.model_name, "split": config.dataset_config.split}},)
            
            with model_artifact.new_file("model_map.pkl", "wb") as f:
                pickle.dump(model, f)
                
            wandb.log_artifact(model_artifact)

        return


if __name__ == "__main__":
    import os
    import sys

    if sys.argv:
        # pass wandb API as argv[1] and set environment variable
        # 'python mll_optim.py MY_API_KEY'
        os.environ["WANDB_API_KEY"] = sys.argv[1]
        
    # Adds jax flags to the program.
    jax.config.config_with_absl()

    # Snippet to pass configs into the main function.
    def _main(argv):
        del argv
        config = FLAGS.config
        main(config)

    app.run(_main)