from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import ml_collections.config_flags
import wandb
from absl import app, flags
from chex import Array

from scalable_gps import kernels
from scalable_gps.data import get_dataset
from scalable_gps.eval_utils import mean_LLH
from scalable_gps.models.cg_gp_model import CGGPModel
from scalable_gps.models.sgd_gp_model import SGDGPModel
from scalable_gps.models.vi_gp_model import SVGPModel, SVGPThompsonInterface
from scalable_gps.utils import (
    HparamsTuple,
    flatten_nested_dict,
    get_clustered_indices,
    get_map_solution,
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

from scalable_gps.thompson_utils import get_acquisition_fn

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

        # Compute stochastic optimised solution
        if config.model_name == "sgd":
            model = SGDGPModel(hparams.noise_scale, kernel)
            train_config = config.train_config
            sampling_config = config.sampling_config
        elif config.model_name == "cg":
            model = CGGPModel(hparams.noise_scale, kernel)
            train_config = config.cg_config
            train_config.preconditioner = False
            sampling_config = config.cg_sampling_config
            sampling_config.preconditioner = False
        elif config.model_name == "precondcg":
            model = CGGPModel(hparams.noise_scale, kernel)
            train_config = config.cg_config
            train_config.preconditioner = True
            sampling_config = config.cg_sampling_config
            sampling_config.preconditioner = True
        elif config.model_name == "vi":
            train_config = config.vi_config
            if config.vi_config.annoy_pre_clustering:
                keep_indices = get_clustered_indices(
                    config.dataset_name,
                    config.dataset_config.split,
                    lengthscale_ratio=config.vi_config.clustering_length_scale_ratio,
                )
                print(f"loaded {len(keep_indices)} keep indices from clustering")
                train_ds.z = train_ds.x[keep_indices]
            model = SVGPModel(hparams.noise_scale, kernel, config)
            model.reinit_get_predictive(train_ds, optim_key)

        metrics_list = ["loss", "err", "reg", "normalised_test_rmse", "test_rmse", "test_llh", "normalised_test_llh"]
        if config.compute_exact_soln:
            metrics_list.extend(["alpha_diff", "y_pred_diff", "alpha_rkhs_diff"])

        try:
            data = get_map_solution(
                config.dataset_name, 
                config.model_name, 
                config.dataset_config.split,
                config.override_noise_scale,
                config.train_config.use_improved_grad
            )
            
            if config.model_name == "vi":
                model.vi_params = data['alpha']
            else:
                model.alpha = data['alpha']
            print('loaded in mean alpha from WANDB server.')
        except:
            model.compute_representer_weights(
                optim_key,
                train_ds,
                test_ds,
                train_config,
                metrics_list=metrics_list,
                metrics_prefix="train",
                exact_metrics=exact_metrics if config.compute_exact_soln else None
            )
        
        if config.model_name != 'vi':
            zero_mean_samples, alpha_samples, _ = model.compute_posterior_samples(
                sampling_key,
                n_samples=config.sampling_config.n_samples,
                train_ds=train_ds,
                test_ds=test_ds,
                config=sampling_config,
                use_rff=True,
                n_features=config.sampling_config.n_features_prior_sample,
                zero_mean=True,
                metrics_list=metrics_list,
                metrics_prefix="sampling",
                compare_exact=False
            )
        else:
            y_pred_loc = model.predictive_mean(train_ds, test_ds)
            
            if not config.vi_config.use_exact_pred_variance:
                # Calculate Inducing points.
                inducing_inputs = model.vi_params["variational_family"]["inducing_inputs"]
                feature_params = kernel.feature_params(key, 2000, inducing_inputs.shape[-1])
                L = kernel.featurise(
                    inducing_inputs,
                    feature_params
                    )
                alpha_map = jnp.zeros(inducing_inputs.shape[0])

                aux_vi_model = SVGPThompsonInterface(hparams.noise_scale, kernel, config)
                aux_vi_model.vi_params = model.vi_params
                _, pseudo_representer_weights, w_samples = aux_vi_model.compute_posterior_samples(
                    sampling_key, config.sampling_config.n_samples, train_ds, test_ds, train_config, L=L)
                
                class DummyState(NamedTuple):
                    feature_params: Array
                
                state = DummyState(feature_params)
                acq_fn, _, _ = get_acquisition_fn(
                    state, kernel, alpha_map, pseudo_representer_weights, 
                    w_samples, inducing_inputs=inducing_inputs)
                
                posterior_variance = acq_fn(test_ds.x)

                y_pred_variance = jnp.var(posterior_variance, axis=0) + hparams.noise_scale ** 2
            else:
                # zero_mean_posterior_samples = model.compute_posterior_samples(
                #     sampling_key, train_ds, test_ds, 64, L=L)
                
                # print(f'zero_mean_posterior_samples: {zero_mean_posterior_samples.shape}')
                y_pred_variance = model.predictive_variance(
                    train_ds, test_ds, add_likelihood_noise=True, return_marginal_variance=True)
            test_llh = mean_LLH(
                test_ds.y, y_pred_loc, y_pred_variance, mu=train_ds.mu_y, sigma=train_ds.sigma_y)
            normalised_test_llh = mean_LLH(test_ds.y, y_pred_loc, y_pred_variance)
            
            wandb.log({"test_llh": test_llh, "normalised_test_llh": normalised_test_llh})
            print(f'test_llh: {test_llh}')
            print(f'normalised_test_llh: {normalised_test_llh}')

        if config.wandb.log_artifact:
            # Use wandb artifacts to save model hparams for a given dataset split and subsample_idx.
            artifact_name = f"samples_{config.dataset_name}_{config.model_name}_{config.dataset_config.split}"
            if config.override_noise_scale > 0.:
                artifact_name += f"_noise_{config.override_noise_scale}"
            if config.sampling_config.use_improved_grad:
                artifact_name += f"_improved_grad"

            samples_artifact = wandb.Artifact(
                artifact_name, type="samples",
                description=f"Saved samples for {config.dataset_name} dataset with method {config.model_name} on split {config.dataset_config.split}.",
                metadata={**{
                    "dataset_name": config.dataset_name,
                    "model_name": config.model_name,
                    "split": config.dataset_config.split,
                    "use_improved_grad": config.sampling_config.use_improved_grad
                    }},
                )
            
            with samples_artifact.new_file("samples.pkl", "wb") as f:
                pickle.dump({'zero_mean_samples': zero_mean_samples, 'alpha_samples': alpha_samples}, f)
            
                
            wandb.log_artifact(samples_artifact)

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