import time
from typing import List, Optional

import chex
import jax.numpy as jnp
import jax.random as jr
import optax
import wandb
from chex import Array
from ml_collections import ConfigDict
from tqdm import tqdm

from scalable_gps import eval_utils, optim_utils, sampling_utils
from scalable_gps.data import Dataset
from scalable_gps.eval_utils import LLH
from scalable_gps.kernels import Kernel
from scalable_gps.models.base_gp_model import GPModel
from scalable_gps.models.exact_gp_model import ExactGPModel
from scalable_gps.optim_utils import get_lr, get_lr_and_schedule
from scalable_gps.utils import (
    ExactPredictionsTuple,
    TargetTuple,
    process_vmapped_metrics,
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
    ):
        """Compute the representer weights alpha by solving alpha = (K + sigma^2 I)^{-1} y using SGD."""
        target_tuple = TargetTuple(error_target=train_ds.y, regularizer_target=jnp.zeros_like(train_ds.y))
        
        optimizer = get_lr_and_schedule(
            "sgd", config, config.lr_schedule_name, config.lr_schedule_config)

        # Define the gradient function
        grad_fn = optim_utils.get_stochastic_gradient_fn(train_ds.x, self.kernel.kernel_fn, self.noise_scale)
        update_fn = optim_utils.get_update_fn(grad_fn, optimizer, config.polyak, vmap=False)
        feature_fn = self.get_feature_fn(train_ds, config.n_features_optim, config.recompute_features)
        
        eval_fn = eval_utils.get_eval_fn(
            metrics_list,
            train_ds,
            test_ds,
            self.kernel.kernel_fn,
            self.noise_scale,
            grad_fn=grad_fn,
            metrics_prefix=metrics_prefix,
            exact_metrics=exact_metrics
        )

        # Initialise alpha and alpha_polyak
        alpha, alpha_polyak = jnp.zeros((train_ds.N,)), jnp.zeros((train_ds.N,))

        opt_state = optimizer.init(alpha)

        idx_key, feature_key = jr.split(key, 2)
        features = feature_fn(feature_key)
        if config.batch_size == 0:
            def partial_fn(batch_size):
                idx_fn = optim_utils.get_idx_fn(batch_size, train_ds.N, config.iterative_idx, share_idx=False)
                idx = idx_fn(0, idx_key)
                update_fn(alpha, alpha_polyak, idx, features, opt_state, target_tuple)
            config.batch_size = optim_utils.select_dynamic_batch_size(train_ds.N, partial_fn)
            print(f"Selected batch size: {config.batch_size}, (N = {train_ds.N}, D = {train_ds.D}, "
                  f"length_scale dims: {self.kernel.get_length_scale().shape[-1]})")
        assert config.batch_size > 0
        idx_fn = optim_utils.get_idx_fn(config.batch_size, train_ds.N, config.iterative_idx, share_idx=False)
        
        # force JIT by running a single step
        # TODO: Wrap this in something we can call outside this function potentially. When we run 10 steps to calculate
        # num_iterations per budget, this will have to be called once there.
        idx = idx_fn(0, idx_key)
        update_fn(alpha, alpha_polyak, idx, features, opt_state, target_tuple)

        wall_clock_time = time.time()
        aux = []
        for i in tqdm(range(config.iterations)):
            start_time = time.time()
            key, idx_key, feature_key = jr.split(key, 3)
            features = feature_fn(feature_key)
            idx = idx_fn(i, idx_key)

            alpha, alpha_polyak, opt_state = update_fn(alpha, alpha_polyak, idx, features, opt_state, target_tuple)
            end_time = time.time()
            wall_clock_time += end_time - start_time
            if i % config.eval_every == 0:
                eval_metrics = eval_fn(alpha_polyak, idx, features, target_tuple)

                lr_to_log = get_lr(opt_state)

                if wandb.run is not None:
                    wandb.log({**eval_metrics, 
                               **{'train_step': i, 'lr': lr_to_log, 'wall_clock_time': wall_clock_time}})
                aux.append(eval_metrics)

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
        metrics_list=[],
        metrics_prefix="",
        compare_exact=False):
        
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
                    chol_eps=chol_eps)
        
        # Get vmapped functions for sampling from the prior and computing the posterior.
        compute_prior_samples_fn = self.get_prior_samples_fn(train_ds.N, L, use_rff)
        compute_posterior_samples_fn = self.get_posterior_samples_fn(train_ds, test_ds, zero_mean)
        compute_target_tuples_fn = optim_utils.get_target_tuples_fn(config.loss_objective)
        
        optimizer = optax.sgd(learning_rate=config.learning_rate, momentum=config.momentum, nesterov=True)
        
        grad_fn = optim_utils.get_stochastic_gradient_fn(train_ds.x, self.kernel.kernel_fn, self.noise_scale)
        update_fn = optim_utils.get_update_fn(grad_fn, optimizer, config.polyak, vmap=True)
        feature_fn = self.get_feature_fn(train_ds, config.n_features_optim, config.recompute_features)
 
        # Call the vmapped functions
        f0_samples_train, f0_samples_test, eps0_samples, w_samples = compute_prior_samples_fn(
            jr.split(prior_samples_key, n_samples))  # (n_samples, n_train), (n_samples, n_test), (n_samples, n_train)

        exact_samples_tuple = None
        if compare_exact:
            exact_gp = ExactGPModel(self.noise_scale, self.kernel)
            exact_gp.K = exact_gp.kernel.kernel_fn(train_ds.x, train_ds.x)
            exact_gp.compute_representer_weights(train_ds)
            
            compute_exact_alpha_samples_fn = exact_gp.get_alpha_samples_fn()
            compute_exact_posterior_samples_fn = exact_gp.get_posterior_samples_fn(train_ds, test_ds, zero_mean=False)
            compute_exact_samples_tuple_fn = eval_utils.get_exact_sample_tuples_fn(exact_gp.alpha)

            alpha_samples_exact = compute_exact_alpha_samples_fn(f0_samples_train, eps0_samples)
            posterior_samples_exact = compute_exact_posterior_samples_fn(alpha_samples_exact, f0_samples_test)

            exact_samples_tuple = compute_exact_samples_tuple_fn(
                alpha_samples_exact, posterior_samples_exact, f0_samples_test)
        
        eval_fn = eval_utils.get_eval_fn(
            metrics_list,
            train_ds,
            test_ds,
            kernel_fn=self.kernel.kernel_fn,
            noise_scale=self.noise_scale,
            grad_fn=grad_fn,
            metrics_prefix=metrics_prefix,
            exact_samples=exact_samples_tuple if compare_exact else None,
            vmap=True
        )

        target_tuples = compute_target_tuples_fn(f0_samples_train, eps0_samples) # (n_samples, TargetTuples)

        alphas, alphas_polyak = jnp.zeros((n_samples, train_ds.N)), jnp.zeros((n_samples, train_ds.N))
        opt_states = optimizer.init(alphas)
        
        idx_key, feature_key = jr.split(key, 2)
        features = feature_fn(feature_key)
        if config.batch_size == 0:
            def partial_fn(batch_size):
                idx_fn = optim_utils.get_idx_fn(batch_size, train_ds.N, config.iterative_idx, share_idx=False)
                idx = idx_fn(0, idx_key)
                update_fn(alphas, alphas_polyak, idx, features, opt_states, target_tuples)
            config.batch_size = optim_utils.select_dynamic_batch_size(train_ds.N, partial_fn)
            print(f"Selected batch size: {config.batch_size}, (N = {train_ds.N}, D = {train_ds.D}, "
                  f"length_scale dims: {self.kernel.get_length_scale().shape[-1]})")
        assert config.batch_size > 0
        idx_fn = optim_utils.get_idx_fn(config.batch_size, train_ds.N, config.iterative_idx, share_idx=False)

        # force JIT
        idx = idx_fn(0, idx_key)
        update_fn(alphas, alphas_polyak, idx, features, opt_states, target_tuples)

        aux = []
        for i in tqdm(range(config.iterations)):
            optim_key, idx_key, feature_key = jr.split(optim_key, 3)
            features = feature_fn(feature_key)

            idx = idx_fn(i, idx_key)

            alphas, alphas_polyak, opt_states = update_fn(alphas, alphas_polyak, idx, features, opt_states, target_tuples)

            if i % config.eval_every == 0:
                vmapped_eval_metrics = eval_fn(alphas_polyak, idx, features, target_tuples)

                aux_metrics = {}
                if "test_llh" in metrics_list or "normalised_test_llh" in metrics_list:
                    y_pred_loc = self.predictive_mean(train_ds, test_ds, recompute=False)
                    zero_mean_posterior_samples = compute_posterior_samples_fn(alphas_polyak, f0_samples_test)
                    y_pred_scale = self.predictive_variance_samples(zero_mean_posterior_samples)
                    del zero_mean_posterior_samples
                    if "test_llh" in metrics_list:
                        aux_metrics['test_llh'] = LLH(
                            test_ds.y, y_pred_loc, y_pred_scale, mu=train_ds.mu_y, sigma=train_ds.sigma_y)
                    if "normalised_test_llh" in metrics_list:
                        aux_metrics['normalised_test_llh'] = LLH(test_ds.y, y_pred_loc, y_pred_scale)
                    del y_pred_loc, y_pred_scale

                if wandb.run is not None:
                    wandb.log({**process_vmapped_metrics(vmapped_eval_metrics),
                            **{'sample_step': i},
                            **aux_metrics})

                aux.append(vmapped_eval_metrics)

        print(f'alphas_polyak: {alphas_polyak.shape}')
        
        posterior_samples = compute_posterior_samples_fn(alphas_polyak, f0_samples_test)  # (n_samples, n_test)
        
        return posterior_samples, alphas_polyak, w_samples, aux