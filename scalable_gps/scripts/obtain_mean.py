"""Calculates posterior mean of a GP model using different methods."""
import jax
import jax.numpy as jnp
import jax.random as jr
import ml_collections.config_flags
import wandb
from absl import app, flags

from scalable_gps import kernels
from scalable_gps.configs.default import get_dataset_config
from scalable_gps.data import get_dataset
from scalable_gps.eval_utils import RMSE, R2_score
from scalable_gps.linear_model import marginal_likelihood
from scalable_gps.models.cg_gp_model import CGGPModel
from scalable_gps.models.exact_gp_model import ExactGPModel
from scalable_gps.models.sgd_gp_model import SGDGPModel
from scalable_gps.models.vi_gp_model import SVGPModel
from scalable_gps.utils import (
    ExactPredictionsTuple,
    HparamsTuple,
    flatten_nested_dict,
    get_clustered_indices,
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

        # Call dataset config again

        if run.config.override_d_name != "":
            new_config = ml_collections.ConfigDict()
            new_config.dataset_config = get_dataset_config(run.config.override_d_name)
            computed_configs = {
                **new_config,
                **{
                    "dataset_config.binarize": run.config.override_d_binarize,
                    "dataset_config.normalise": True,
                    "dataset_config.split": 0,
                    "dataset_name": run.config.override_d_name,
                },
            }
        else:
            computed_configs = {}
        computed_configs["train_config.polyak"] = (
            100 / run.config["train_config.iterations"]
        )
        update_config_dict(config, run, computed_configs)

        print(config)

        # Obtain Dataset and HParams
        train_ds, test_ds = get_dataset(config.dataset_name, **config.dataset_config)

        print(f"N: {train_ds.N}, D: {train_ds.D}, N_test: {test_ds.N}")
        print(f"x_train: {train_ds.x.shape}, y_train: {train_ds.y.shape}")

        try:
            hparams = get_tuned_hparams(
                config.dataset_name, config.dataset_config.split
            )
        except wandb.CommError:
            print("Could not fetch hparams from wandb. Using default values.")

            hparams = HparamsTuple(
                length_scale=jnp.array(config.kernel_config.length_scale),
                signal_scale=config.kernel_config.signal_scale,
                noise_scale=config.dataset_config.noise_scale,
            )
        if config.override_noise_scale > 0.0:
            hparams = HparamsTuple(
                length_scale=hparams.length_scale,
                signal_scale=hparams.signal_scale,
                noise_scale=config.override_noise_scale,
            )
        print(hparams)

        # Initialise Kernel
        kernel_init_fn = getattr(kernels, config.kernel_name)
        kernel = kernel_init_fn(
            {"signal_scale": hparams.signal_scale, "length_scale": hparams.length_scale}
        )

        key = jr.PRNGKey(config.seed)
        optim_key, sampling_key, key = jr.split(key, 3)

        # Compute exact solution
        exact_model, exact_metrics = None, None
        if config.compute_exact_soln:
            exact_model = ExactGPModel(hparams.noise_scale, kernel)

            exact_model.compute_representer_weights(train_ds)

            print(exact_model.alpha)
            y_pred_exact = exact_model.predictive_mean(train_ds, test_ds)
            test_rmse_exact = RMSE(
                test_ds.y, y_pred_exact, mu=train_ds.mu_y, sigma=train_ds.sigma_y
            )
            normalised_test_rmse = RMSE(test_ds.y, y_pred_exact)

            mll = marginal_likelihood(
                train_ds.x, train_ds.y, exact_model.kernel.kernel_fn, hparams
            )

            exact_r2 = R2_score(
                test_ds.y, y_pred_exact, mu=train_ds.mu_y, sigma=train_ds.sigma_y
            )

            print(f"test_rmse_exact = {test_rmse_exact}")
            print(f"r2_exact = {exact_r2}")
            wandb.log(
                {
                    "exact/test_rmse": test_rmse_exact,
                    "exact/normalised_test_rmse": normalised_test_rmse,
                    "exact/mll": mll / train_ds.N,
                    "exact/r2": exact_r2,
                }
            )

            # Define exact metrics that we will use later to compare with stochastic solution
            exact_metrics = ExactPredictionsTuple(
                alpha=exact_model.alpha, y_pred_loc=y_pred_exact
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
            model = SVGPModel(hparams.noise_scale, kernel, config)

            if config.vi_config.annoy_pre_clustering:
                keep_indices = get_clustered_indices(
                    config.dataset_name,
                    config.dataset_config.split,
                    lengthscale_ratio=config.vi_config.clustering_length_scale_ratio,
                )
                print(f"loaded {len(keep_indices)} keep indices from clustering")
                train_ds.z = train_ds.x[keep_indices]

        # metrics_list = ["loss", "err", "reg", "normalised_test_rmse", "test_rmse"]
        metrics_list = ["normalised_test_rmse", "test_rmse", "R2"]
        if config.compute_exact_soln:
            metrics_list.extend(
                ["alpha_diff", "alpha_rkhs_diff", "y_pred_diff", "y_pred_test_diff"]
            )

        # Compute the SGD MAP solution for representer weights.

        artifact_name = f"alpha_{config.dataset_name}_{config.model_name}_{config.dataset_config.split}"
        if config.override_noise_scale > 0.0:
            artifact_name += f"_noise_{config.override_noise_scale}"

        alpha, aux = model.compute_representer_weights(
            optim_key,
            train_ds,
            test_ds,
            train_config,
            metrics_list=metrics_list,
            metrics_prefix="train",
            exact_metrics=exact_metrics if config.compute_exact_soln else None,
            artifact_name=artifact_name,
        )

        y_pred = model.predictive_mean(train_ds, test_ds)
        test_rmse = RMSE(test_ds.y, y_pred, mu=train_ds.mu_y, sigma=train_ds.sigma_y)
        normalised_test_rmse = RMSE(test_ds.y, y_pred)

        print("test_rmse = ", test_rmse)
        print("normalised_test_rmse = ", normalised_test_rmse)
        wandb.log(
            {"test_rmse": test_rmse, "normalised_test_rmse": normalised_test_rmse}
        )

        if config.wandb.log_artifact:
            # Use wandb artifacts to save model hparams for a given dataset split and subsample_idx.
            artifact_name = f"alpha_{config.dataset_name}_{config.model_name}_{config.dataset_config.split}"
            if config.override_noise_scale > 0.0:
                artifact_name += f"_noise_{config.override_noise_scale}"
            artifact_name += f"_{config.train_config.grad_variant}_{config.train_config.learning_rate}"
            model_artifact = wandb.Artifact(
                artifact_name,
                type="alpha",
                description=f"Saved alpha for {config.dataset_name} dataset with method {config.model_name} on split {config.dataset_config.split}.",
                metadata={
                    **{
                        "dataset_name": config.dataset_name,
                        "model_name": config.model_name,
                        "split": config.dataset_config.split,
                        "grad_variant": config.train_config.grad_variant,
                        "learning_rate": config.train_config.learning_rate,
                    }
                },
            )

            with model_artifact.new_file("alpha_map.pkl", "wb") as f:
                pickle.dump({"alpha": alpha, "aux": aux}, f)

            wandb.log_artifact(model_artifact)

        return


if __name__ == "__main__":
    import os

    # Adds jax flags to the program.
    jax.config.config_with_absl()

    # Snippet to pass configs into the main function.
    def _main(argv):
        del argv
        config = FLAGS.config
        os.environ["WANDB_API_KEY"] = config.wandb.api_key
        main(config)

    app.run(_main)
