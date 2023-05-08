
import time
from typing import Optional

import gpjax as gpx
import jax.numpy as jnp
import ml_collections
import optax
import wandb
from chex import Array

from scalable_gps.data import Dataset
from scalable_gps.kernels import Kernel
from scalable_gps.SVGP import regression_SVGP


class SVGPModel:
    def __init__(self, noise_scale: float, kernel: Kernel, config: ml_collections.ConfigDict, kernel_config: dict):
        self.alpha = None
        self.y_pred = None
        self.K = None
        self.kernel = kernel
        self.noise_scale = noise_scale
        
        self.regression_fn = self.get_regression_fn(
            num_inducing=config.vi_config.num_inducing_points, 
            kernel_name=config.kernel_name, 
            kernel_config=kernel_config, 
            ARD=config.kernel_config.use_ard, 
            noise_scale=noise_scale, 
            inducing_init=config.vi_config.inducing_init,)
    

    def get_regression_fn(self, num_inducing, kernel_name, kernel_config, ARD, noise_scale, inducing_init):
        
        def _fn(ds, key):
            return regression_SVGP(
                ds,
                num_inducing=num_inducing,
                kernel_name=kernel_name,
                kernel_config=kernel_config,
                ARD=ARD,
                noise_scale=noise_scale,
                key=key,
                inducing_init=inducing_init,)

        return _fn


    def compute_representer_weights(
        self, 
        key, 
        train_ds, 
        test_ds, 
        config: ml_collections.ConfigDict,
        metrics_list: Optional[list]=None,
        metrics_prefix: Optional[str]=None,
        exact_metrics: Optional[list]=None,):
        
        del metrics_list, metrics_prefix, exact_metrics
        
        optimizer = optax.adam(learning_rate=config.learning_rate)
        absolute_clipping = config.absolute_clipping
        optimizer = optax.chain(optax.zero_nans(), optax.clip_by_global_norm(absolute_clipping), optimizer)

        negative_elbo, init_state, D, self.get_predictive = self.regression_fn(train_ds, key)
        
        start_time = time.time()
        optimised_state = gpx.fit_batches(
            objective=negative_elbo,
            parameter_state=init_state,
            train_data=D,
            optax_optim=optimizer,
            num_iters=config.iterations,
            key=key,
            batch_size=config.batch_size,
            verbose=False
        )
        
        self.vi_params, loss = optimised_state.unpack()
        
        self.function_dist, self.predictive_dist = self.get_predictive(
            self.vi_params, test_ds.x)
        
        end_time = time.time()
        
        if wandb.run is not None:
            for loss_val in loss:
                wandb.log({"loss": loss_val})
            wandb.log({"wall_clock_time": end_time - start_time})
        
        return self.vi_params, (self.function_dist, self.predictive_dist)
        

    def predictive_mean(self, train_ds: Dataset, test_ds: Dataset, recompute: bool = True) -> Array:
        if self.predictive_dist is None or self.function_dist is None and not recompute:
            raise ValueError("vi_params is None. Please call compute_representer_weights() first.")
        
        if recompute:
            self.function_dist, self.predictive_dist = self.get_predictive(
                self.vi_params, test_ds.x)
        
        
        self.y_pred = self.predictive_dist.mean()

        return self.y_pred  # (N_test, 1)

    def predictive_variance(
        self, train_ds: Dataset, test_ds: Dataset, return_marginal_variance: bool = True, recompute: bool=False) -> Array:
        """Compute the posterior variance of the test points."""
        if self.predictive_dist is None or self.function_dist is None and not recompute:
            raise ValueError("vi_params is None. Please call compute_representer_weights() first.")
        
        if recompute:
            self.function_dist, self.predictive_dist = self.get_predictive(
                self.vi_params, test_ds.x)
        
        variance  = self.predictive_dist.variance()
        
        # TODO: is this correct?
        if return_marginal_variance:
            return jnp.diag(variance)
        else:
            return variance

    def compute_posterior_samples(self, key, num_samples):
        posterior_samples = self.function_dist.sample(seed=key, sample_shape=(num_samples, ))
        
        zero_mean_posterior_samples = posterior_samples - self.predictive_dist.mean()
        
        return zero_mean_posterior_samples


    # TODO: Biased: use the method that double counts the diagonal (first paragraph of page 28 of https://arxiv.org/pdf/2210.04994.pdf
    # TODO: Unbiased: use a mixture of isotropic Gaussian likelihood with each mixture component's mean being centred at a sample. Then we can compute joint likelihoods as in the "EFFICIENT κ-ADIC SAMPLING" section on page 26 of https://arxiv.org/pdf/2210.04994.pdf
    def predictive_variance_samples(
        self, zero_mean_posterior_samples: Array, return_marginal_variance: bool = True) -> Array:
        """Compute MC estimate of posterior variance of the test points using zero mean samples from posterior."""
        if self.predictive_dist is None or self.function_dist is None and not recompute:
            raise ValueError("vi_params is None. Please call compute_representer_weights() first.")
        
        if return_marginal_variance:
            variance = jnp.mean(zero_mean_posterior_samples ** 2, axis=0)  # (N_test, 1)
            variance -= self.noise_scale ** 2
        else:
            n_samples = zero_mean_posterior_samples.shape[0]
            variance = zero_mean_posterior_samples.T @ zero_mean_posterior_samples / n_samples
        
            variance -= (self.noise_scale ** 2) * jnp.eye(variance.shape[0])
        
        return variance