import time
from typing import List, Optional

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import wandb
from chex import Array
from ml_collections import ConfigDict
from tqdm import tqdm

from scalable_gps import eval_utils, optim_utils, sampling_utils
from scalable_gps.data import Dataset
from scalable_gps.eval_utils import mean_LLH
from scalable_gps.kernels import Kernel
from scalable_gps.models.base_gp_model import GPModel
from scalable_gps.models.exact_gp_model import ExactGPModel
from scalable_gps.optim_utils import get_lr, get_lr_and_schedule
from scalable_gps.utils import (
    ExactPredictionsTuple,
    TargetTuple,
    get_latest_saved_artifact,
    process_pmapped_and_vmapped_metrics,
    save_latest_artifact,
)


class SGDGPModel(GPModel):
    def __init__(self, noise_scale: float, kernel: Kernel, **kwargs):
        super().__init__(noise_scale=noise_scale, kernel=kernel, **kwargs)

    def compute_representer_weights(
        self,
        key: chex.PRNGKey,
        train_ds: Dataset,
        test_ds: Dataset,
        config: ConfigDict,
        metrics_list: List[str],
        metrics_prefix: str = "",
        exact_metrics: Optional[ExactPredictionsTuple] = None,
        recompute: Optional[bool] = None,
        artifact_name: Optional[str] = None,
    ):
        del recompute
        """Compute the representer weights alpha by solving alpha = (K + sigma^2 I)^{-1} y using SGD."""
        target_tuple = TargetTuple(
            error_target=train_ds.y, regularizer_target=jnp.zeros_like(train_ds.y)
        )

        optimizer = get_lr_and_schedule(
            "sgd", config, config.lr_schedule_name, config.lr_schedule_config
        )

        # Define the gradient function
        grad_fn = optim_utils.get_stochastic_gradient_fn(train_ds.x, self.kernel.kernel_fn, self.noise_scale, config.use_improved_grad)
        update_fn = optim_utils.get_update_fn(grad_fn, optimizer, config.polyak)

        if config.use_improved_grad:
            feature_fn = lambda _: None
        else:
            feature_fn = self.get_feature_fn(train_ds, config.n_features_optim, config.recompute_features)
        
        eval_fn = eval_utils.get_eval_fn(
            metrics_list,
            train_ds,
            test_ds,
            self.kernel.kernel_fn,
            self.noise_scale,
            grad_fn=grad_fn,
            metrics_prefix=metrics_prefix,
            exact_metrics=exact_metrics,
        )

        # Initialise alpha and alpha_polyak
        alpha, alpha_polyak = jnp.zeros((train_ds.N,)), jnp.zeros((train_ds.N,))

        opt_state = optimizer.init(alpha)

        idx_key, feature_key = jr.split(key, 2)
        features = feature_fn(feature_key)
        if config.batch_size == 0:

            def partial_fn(batch_size):
                idx_fn = optim_utils.get_idx_fn(
                    batch_size, train_ds.N, config.iterative_idx, share_idx=False
                )
                idx = idx_fn(0, idx_key)
                update_fn(alpha, alpha_polyak, idx, features, opt_state, target_tuple)

            config.batch_size = optim_utils.select_dynamic_batch_size(
                train_ds.N, partial_fn
            )
            print(
                f"Selected batch size: {config.batch_size}, (N = {train_ds.N}, D = {train_ds.D}, "
                f"length_scale dims: {self.kernel.get_length_scale().shape[-1]})"
            )
        assert config.batch_size > 0
        idx_fn = optim_utils.get_idx_fn(
            config.batch_size, train_ds.N, config.iterative_idx, share_idx=False
        )

        # force JIT by running a single step
        # TODO: Wrap this in something we can call outside this function potentially. When we run 10 steps to calculate
        # num_iterations per budget, this will have to be called once there.
        idx = idx_fn(0, idx_key)
        update_fn(alpha, alpha_polyak, idx, features, opt_state, target_tuple)
        
        wall_clock_time = 0.
        
        ########### ADD PREEMPTIBLE SAFE CKPT LOADING AND SAVING ####################
        all_save_steps = list(jnp.arange(0, config.iterations, config.iterations // 10).astype(int))[1:]
        print(f'All save steps: {all_save_steps}')
        most_recent_artifact_data = None
        if config.preempt_safe:
            most_recent_artifact_data = get_latest_saved_artifact(
                artifact_name, all_save_steps)
            print(f'Most recent artifact data: {most_recent_artifact_data}')
        
        if most_recent_artifact_data is not None:
            restart_step = most_recent_artifact_data["train_step"]
            alpha = most_recent_artifact_data["alpha"]
            alpha_polyak = most_recent_artifact_data["alpha_polyak"]
            opt_state = most_recent_artifact_data["opt_state"]

            wall_clock_time = most_recent_artifact_data["wall_clock_time"]
        
    

        aux = []
        for i in tqdm(range(config.iterations)):
            if most_recent_artifact_data is not None and i < restart_step:
                continue
            start_time = time.time()
            key, idx_key, feature_key = jr.split(key, 3)
            features = feature_fn(feature_key)
            idx = idx_fn(i, idx_key)
            alpha, alpha_polyak, opt_state = update_fn(alpha, alpha_polyak, idx, features, opt_state, target_tuple)
            alpha.block_until_ready()
            end_time = time.time()
            wall_clock_time += end_time - start_time
            if i % config.eval_every == 0:
                eval_metrics = eval_fn(alpha_polyak, idx, features, target_tuple)

                lr_to_log = get_lr(opt_state)

                if wandb.run is not None:
                    wandb.log(
                        {
                            **eval_metrics,
                            **{
                                "train_step": i,
                                "lr": lr_to_log,
                                "wall_clock_time": wall_clock_time,
                            },
                        }
                    )
                aux.append(eval_metrics)
            
            if config.preempt_safe and i % (config.iterations // 10) == 0 and i > 0:
                artifact_data = {
                    'alpha': alpha,
                    'alpha_polyak': alpha_polyak,
                    'opt_state': opt_state,
                    'train_step': i,
                    'wall_clock_time': wall_clock_time}
                save_artifact_name = f"{artifact_name}_{i}"
                print(f'Saving artifact at step {i}, {save_artifact_name}')
                save_latest_artifact(artifact_data, save_artifact_name)

        self.alpha = alpha_polyak
        return self.alpha, aux

    def compute_posterior_samples(
        self,
        key: chex.PRNGKey,
        n_samples: int,
        train_ds: Dataset,
        test_ds: Dataset,
        config: ConfigDict,
        use_rff: bool = True,
        n_features: int = 0,
        chol_eps: float = 1e-5,
        L: Optional[Array] = None,
        zero_mean: bool = True,
        metrics_list: list = [],
        metrics_prefix: str = "",
        compare_exact: bool = False):
        
        prior_covariance_key, prior_samples_key, optim_key = jr.split(key, 3)

        if L is None:
            L = sampling_utils.compute_prior_covariance_factor(
                prior_covariance_key,
                train_ds,
                test_ds,
                self.kernel.kernel_fn,
                self.kernel.feature_fn,
                use_rff=use_rff,
                n_features=n_features,
                chol_eps=chol_eps,
            )

        # Get vmapped functions for sampling from the prior and computing the posterior.
        compute_prior_samples_fn = self.get_prior_samples_fn(train_ds.N, L, use_rff, pmap=True)
        compute_posterior_samples_fn = self.get_posterior_samples_fn(train_ds, test_ds, zero_mean, pmap=True)
        compute_target_tuples_fn = optim_utils.get_target_tuples_fn(config.loss_objective, pmap=True)
        
        optimizer = optax.sgd(learning_rate=config.learning_rate, momentum=config.momentum, nesterov=config.nesterov)
        
        grad_fn = optim_utils.get_stochastic_gradient_fn(train_ds.x, self.kernel.kernel_fn, self.noise_scale, config.use_improved_grad)
        update_fn = optim_utils.get_update_fn(grad_fn, optimizer, config.polyak, vmap_and_pmap=True)

        if config.use_improved_grad:
            feature_fn = lambda _: None
        else:
            feature_fn = self.get_feature_fn(train_ds, config.n_features_optim, config.recompute_features)
 
        # Call the pmapped and vmapped functions
        manual_split_size = 20
        
        n_devices = jax.device_count()
        assert n_samples % n_devices == 0
        n_samples_per_device = n_samples // n_devices
        pmappable_keys = jr.split(
            prior_samples_key, manual_split_size * n_samples
        ).reshape((manual_split_size, n_devices, n_samples_per_device, -1))
        # (n_devices, n_samples_per_device, n_train), (n_devices, n_samples_per_device, n_test)
        
        w_samples = []
        f0_samples_train = jnp.zeros(
            (n_devices, n_samples_per_device, train_ds.x.shape[0])
        )
        f0_samples_test = jnp.zeros(
            (n_devices, n_samples_per_device, test_ds.x.shape[0])
        )

        for ii, L_i in enumerate(jnp.split(L, manual_split_size, axis=1)):
            compute_prior_samples_fn = self.get_prior_samples_fn(
                train_ds.N, L_i, use_rff, pmap=True
            )
            # (n_devices, n_samples_per_device, n_train), (n_devices, n_samples_per_device, n_test)
            pmappable_keys_ = pmappable_keys[ii]
            (
                f0_samples_train_,
                f0_samples_test_,
                eps0_samples,
                w_samples_,
            ) = compute_prior_samples_fn(pmappable_keys_)
            w_samples.append(w_samples_)
            f0_samples_train += f0_samples_train_
            f0_samples_test += f0_samples_test_

        del L
        w_samples = jnp.concatenate(w_samples, axis=-1)
        
        
        exact_samples_tuple = None
        if compare_exact:
            exact_gp = ExactGPModel(self.noise_scale, self.kernel)
            exact_gp.K = exact_gp.kernel.kernel_fn(train_ds.x, train_ds.x)
            exact_gp.compute_representer_weights(train_ds)

            # Reshape from (n_devices, n_samples_per_device, n_train) to (n_samples, n_train)
            f0_samples_train_reshaped = jax.device_put(
                f0_samples_train.reshape(n_samples, train_ds.N), jax.devices('cpu')[0])
            eps0_samples_reshaped = jax.device_put(
                eps0_samples.reshape(n_samples, train_ds.N), jax.devices('cpu')[0])
            f0_samples_test_reshaped = jax.device_put(
                f0_samples_test.reshape(n_samples, test_ds.N), jax.devices('cpu')[0])

            compute_exact_alpha_samples_fn = exact_gp.get_alpha_samples_fn()
            compute_exact_posterior_samples_fn = exact_gp.get_posterior_samples_fn(train_ds, test_ds, zero_mean=False)
            compute_exact_samples_tuple_fn = eval_utils.get_exact_sample_tuples_fn(exact_gp.alpha)

            alpha_samples_exact = compute_exact_alpha_samples_fn(f0_samples_train_reshaped, eps0_samples_reshaped)
            posterior_samples_exact = compute_exact_posterior_samples_fn(alpha_samples_exact, f0_samples_test_reshaped)
            exact_samples_tuple = compute_exact_samples_tuple_fn(alpha_samples_exact, posterior_samples_exact, f0_samples_test_reshaped)
        
        eval_fn = eval_utils.get_eval_fn(
            metrics_list,
            train_ds,
            test_ds,
            kernel_fn=self.kernel.kernel_fn,
            noise_scale=self.noise_scale,
            grad_fn=grad_fn,
            metrics_prefix=metrics_prefix,
            exact_samples=exact_samples_tuple if compare_exact else None,
            vmap_and_pmap=True
        )

        target_tuples = compute_target_tuples_fn(f0_samples_train, eps0_samples) # (n_devices, n_samples_per_device, TargetTuples)

        alphas = jnp.zeros((n_devices, n_samples_per_device, train_ds.N))
        alphas_polyak = jnp.zeros((n_devices, n_samples_per_device, train_ds.N))

        opt_states = optimizer.init(alphas)

        idx_key, feature_key = jr.split(key, 2)
        features = feature_fn(feature_key)
        # if config.batch_size == 0:
        #     def partial_fn(batch_size):
        #         idx_fn = optim_utils.get_idx_fn(batch_size, train_ds.N, config.iterative_idx, share_idx=False)
        #         idx = idx_fn(0, idx_key)
        #         update_fn(alphas, alphas_polyak, idx, features, opt_states, target_tuples)
        #     config.batch_size = optim_utils.select_dynamic_batch_size(train_ds.N, partial_fn)
        #     print(f"Selected batch size: {config.batch_size}, (N = {train_ds.N}, D = {train_ds.D}, "
        #           f"length_scale dims: {self.kernel.get_length_scale().shape[-1]})")
        # assert config.batch_size > 0
        idx_fn = optim_utils.get_idx_fn(config.batch_size, train_ds.N, config.iterative_idx, share_idx=False)

        # # force JIT
        idx = idx_fn(0, idx_key)
        update_fn(alphas, alphas_polyak, idx, features, opt_states, target_tuples)

        aux = []
        for i in tqdm(range(config.iterations)):
            optim_key, idx_key, feature_key = jr.split(optim_key, 3)
            features = feature_fn(feature_key)

            idx = idx_fn(i, idx_key)

            # 0, 0, None, None, 0, 0
            alphas, alphas_polyak, opt_states = update_fn(alphas, alphas_polyak, idx, features, opt_states, target_tuples)

            if i % config.eval_every == 0:
                pmapped_and_vmapped_eval_metrics = eval_fn(alphas_polyak, idx, features, target_tuples)

                aux_metrics = {}
                if "test_llh" in metrics_list or "normalised_test_llh" in metrics_list:
                    y_pred_loc = self.predictive_mean(train_ds, test_ds, recompute=False)
                    zero_mean_posterior_samples = compute_posterior_samples_fn(alphas_polyak, f0_samples_test)
                    y_pred_variance = self.predictive_variance_samples(zero_mean_posterior_samples.reshape(n_samples, test_ds.N), add_likelihood_noise=True)
                    del zero_mean_posterior_samples
                    if "test_llh" in metrics_list:
                        aux_metrics['test_llh'] = mean_LLH(
                            test_ds.y, y_pred_loc, y_pred_variance, mu=train_ds.mu_y, sigma=train_ds.sigma_y)
                    if "normalised_test_llh" in metrics_list:
                        aux_metrics['normalised_test_llh'] = mean_LLH(test_ds.y, y_pred_loc, y_pred_variance)
                    del y_pred_loc, y_pred_variance

                if wandb.run is not None:
                    wandb.log({**process_pmapped_and_vmapped_metrics(pmapped_and_vmapped_eval_metrics),
                            **{'sample_step': i},
                            **aux_metrics})

                aux.append(pmapped_and_vmapped_eval_metrics)

        # print(f"alphas_polyak: {alphas_polyak.shape}")

        posterior_samples = compute_posterior_samples_fn(
            alphas_polyak, f0_samples_test
        )  # (n_samples, n_test)

        return posterior_samples, alphas_polyak, w_samples
