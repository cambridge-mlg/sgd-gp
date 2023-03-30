from functools import partial
from typing import List, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import ml_collections
import optax
import optim_utils
import sampling_utils
from chex import Array
from data import Dataset
from kernels import Kernel
from linalg_utils import KvP, solve_K_inv_v
from linear_model import (
    loss_fn,
)
from metrics import RMSE
from tqdm import tqdm
from train_utils import (
    get_eval_fn,
    get_stochastic_gradient_fn,
    get_update_fn,
    train,
)
from utils import ExactMetricsTuple, TargetTuple


class GPModel:
    def __init__(self, noise_scale: float, kernel: Kernel):
        self.alpha = None
        self.K = None
        self.kernel = kernel
        self.noise_scale = noise_scale

    def compute_representer_weights(self):
        raise NotImplementedError("compute_representer_weights() must be implemented in the derived class.")

    def predictive_mean(self, train_ds: Dataset, test_ds: Dataset) -> Array:
        if self.alpha is None:
            raise ValueError("alpha is None. Please call compute_representer_weights() first.")
        y_pred = KvP(test_ds.x, train_ds.x, self.alpha, kernel_fn=self.kernel.kernel_fn).squeeze() # TODO: Can we remove the squeeze?

        return y_pred  # (N_test, 1)

    def predictive_variance_samples(
        self, zero_mean_posterior_samples: Array, return_marginal_variance: bool = True) -> Array:
        """Compute MC estimate of posterior variance of the test points using zero mean samples from posterior."""
        # zero_mean_posterior_samples = (N_samples, N_test)
        if return_marginal_variance:
            variance = jnp.mean(zero_mean_posterior_samples ** 2, axis=0)  # (N_test, 1)
        else:
            n_samples = zero_mean_posterior_samples.shape[0]
            variance = zero_mean_posterior_samples.T @ zero_mean_posterior_samples / n_samples
    
        return variance

    def calculate_test_rmse(self, train_ds: Dataset, test_ds: Dataset) -> Tuple[Array, Array]:
        y_pred = self.predictive_mean(train_ds, test_ds)
        test_rmse = RMSE(y_pred, test_ds.y, mu=train_ds.mu_y, sigma=train_ds.sigma_y)

        return test_rmse, y_pred
    
    # vmap factory function for sampling from the prior.
    def get_prior_samples_fn(self, n_train, L, use_rff: bool=False):
        # fn(keys) -> prior_samples
        def _fn(key):
            prior_sample_key, sample_key = jr.split(key)
            f0_sample_train, f0_sample_test = sampling_utils.draw_f0_sample(
                    prior_sample_key, n_train, L, use_rff=use_rff)
            eps0_sample = sampling_utils.draw_eps0_sample(
                sample_key, n_train, noise_scale=self.noise_scale)

            return f0_sample_train, f0_sample_test, eps0_sample

        return jax.jit(jax.vmap(_fn))

    # Vmap factory function for computing the zero mean posterior from sample.
    def get_posterior_samples_fn(self, train_ds, test_ds, zero_mean: bool = True):

        def _fn(alpha_sample, f0_sample_test):
            zero_mean_posterior_sample = sampling_utils.compute_posterior_fn_sample(
                train_ds, 
                test_ds, 
                alpha_sample, 
                self.alpha, 
                f0_sample_test, 
                self.kernel.kernel_fn, 
                zero_mean=zero_mean)

            return zero_mean_posterior_sample

        return jax.jit(jax.vmap(_fn))

    def get_feature_fn(self, train_ds: Dataset, n_features: int, recompute: bool):
        
        def _fn(key):
            return self.kernel.feature_fn(
                key, 
                n_features=n_features, 
                recompute=recompute, 
                x=train_ds.x)
        
        return jax.jit(_fn)

class ExactGPModel(GPModel):

    def compute_representer_weights(self, train_ds: Dataset) -> Array:
        """Compute the representer weights alpha by solving alpha = (K + sigma^2 I)^{-1} y"""

        # Compute Kernel exactly
        self.K = self.kernel.kernel_fn(train_ds.x, train_ds.x) if self.K is None else self.K

        # Compute the representer weights by solving alpha = (K + sigma^2 I)^{-1} y
        self.alpha = solve_K_inv_v(self.K, train_ds.y, noise_scale=self.noise_scale)

        return self.alpha

    def predictive_variance(
        self, train_ds: Dataset, test_ds: Dataset, return_marginal_variance: bool = True) -> Array:
        """Compute the posterior variance of the test points."""
        K_test = self.kernel.kernel_fn(test_ds.x, test_ds.x)  # N_test, N_test
        K_train_test = self.kernel.kernel_fn(train_ds.x, test_ds.x)  # N_train, N_test
        # Compute Kernel exactly
        self.K = self.kernel.kernel_fn(train_ds.x, train_ds.x) if self.K is None else self.K

        K_inv_K_train_test = solve_K_inv_v(self.K, K_train_test, noise_scale=self.noise_scale)
    
        variance = K_test - K_train_test.T @ K_inv_K_train_test
        
        if return_marginal_variance:
            return jnp.diag(variance)  # (N_test, 1)
        return variance  # (N_test, N_test)
    
    def get_alpha_samples_fn(self):
        
        def _fn(f0_sample_train, eps0_sample):
            alpha_sample = solve_K_inv_v(
                self.K,
                f0_sample_train + eps0_sample,
                noise_scale=self.noise_scale,
            )
            return alpha_sample

        return jax.jit(jax.vmap(_fn))
    
    def compute_posterior_samples(
        self, 
        key: chex.PRNGKey, 
        n_samples: int, 
        train_ds: Dataset, 
        test_ds: Dataset, 
        use_rff: bool = True,
        n_features: int = 0,
        chol_eps: float = 1e-5,
        L: Optional[Array] = None, 
        zero_mean: bool = True):

        prior_covariance_key, prior_samples_key, samples_optim_key = jr.split(key, 3)
    
        if L is None:
            L, K_train = sampling_utils.compute_prior_covariance_factor(
                prior_covariance_key, 
                train_ds, 
                test_ds, 
                self.kernel.kernel_fn, 
                self.kernel.feature_fn,
                use_rff=use_rff, 
                n_features=n_features, 
                chol_eps=chol_eps)
            self.K = K_train if self.K is None else self.K  # TODO: Do we need to do this?

        # Get vmapped functions for sampling from the prior and computing the posterior.
        compute_prior_samples_fn = self.get_prior_samples_fn(train_ds.N, L, use_rff)
        compute_alpha_samples_fn = self.get_alpha_samples_fn()
        compute_posterior_samples_fn = self.get_posterior_samples_fn(train_ds, test_ds, zero_mean)

        # Call the vmapped functions
        f0_samples_train, f0_samples_test, eps0_samples = compute_prior_samples_fn(
            jr.split(prior_samples_key, n_samples))  # (n_samples, n_train), (n_samples, n_test), (n_samples, n_train)

        alpha_samples = compute_alpha_samples_fn(f0_samples_train, eps0_samples)  # (n_samples, n_train)
        
        posterior_samples = compute_posterior_samples_fn(alpha_samples, f0_samples_test)  # (n_samples, n_test)

        return posterior_samples, alpha_samples


class SGDGPModel(GPModel):
    def __init__(self, noise_scale: float, kernel: Kernel, **kwargs):
        super().__init__(noise_scale=noise_scale, kernel=kernel, **kwargs)

    def compute_representer_weights(
        self,
        key: chex.PRNGKey,
        train_ds: Dataset,
        test_ds: Dataset,
        config: ml_collections.ConfigDict,
        metrics: List[str],
        metrics_prefix: str = "",
        exact_metrics: Optional[ExactMetricsTuple] = None,
    ):
        """Compute the representer weights alpha by solving alpha = (K + sigma^2 I)^{-1} y using SGD."""
        target_tuple = TargetTuple(train_ds.y, jnp.zeros_like(train_ds.y))
        
        # Define the gradient function
        grad_fn = get_stochastic_gradient_fn(
            x=train_ds.x,
            target_tuple=target_tuple,
            kernel_fn=self.kernel.kernel_fn,
            noise_scale=self.noise_scale,
        )

        # Define the gradient update function
        update_fn = get_update_fn(
            grad_fn=grad_fn, 
            n_train=train_ds.N, 
            polyak_step_size=config.polyak)

        feature_fn = partial(
            self.kernel.feature_fn, 
            n_features=config.n_features)
        
        eval_fn = get_eval_fn(
            metrics,
            train_ds,
            test_ds,
            loss_fn,
            grad_fn,
            target_tuple,
            kernel_fn=self.kernel.kernel_fn,
            feature_fn=feature_fn,
            noise_scale=self.noise_scale,
            metrics_prefix=metrics_prefix,
            exact_metrics=exact_metrics
        )

        # Initialise alpha and alpha_polyak
        init_alpha, init_alpha_polyak = self._init_params(train_ds)

        # Solve optimisation problem to obtain representer_weights
        self.alpha, aux = train(
            key, config, update_fn, eval_fn, feature_fn, train_ds, init_alpha, init_alpha_polyak
        )

        return self.alpha, aux


    def compute_posterior_samples( 
        self, 
        key: chex.PRNGKey, 
        n_samples: int,
        train_ds: Dataset, 
        test_ds: Dataset, 
        config: ml_collections.ConfigDict,
        use_rff: bool = True,
        n_features: int = 0,
        chol_eps: float = 1e-5,
        L: Optional[Array] = None, 
        zero_mean: bool = True,
        metrics=[],
        metrics_prefix="",
        compare_exact=False):
        
        prior_covariance_key, prior_samples_key, samples_optim_key = jr.split(key, 3)
    
        L, K_train = L, K_train = sampling_utils.compute_prior_covariance_factor(
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
        idx_fn = optim_utils.get_idx_fn(config.batch_size, config.n_train)
        # get_eval_fn(
        #     metrics,
        #     train_ds,
        #     test_ds,
        #     loss_fn,
        #     grad_fn,
        #     kernel_fn=self.kernel.kernel_fn,
        #     feature_fn=feature_fn,
        #     noise_scale=self.noise_scale,
        #     metrics_prefix=metrics_prefix,
        #     exact_samples=exact_samples,
        # )

        # Call the vmapped functions
        f0_samples_train, f0_samples_test, eps0_samples = compute_prior_samples_fn(
            jr.split(prior_samples_key, n_samples))  # (n_samples, n_train), (n_samples, n_test), (n_samples, n_train)
        
        
        if compare_exact:
            exact_gp = ExactGPModel(self.noise_scale, self.kernel)
            exact_gp.K = K_train
            exact_gp.compute_representer_weights(train_ds)
            
            compute_exact_alpha_samples_fn = exact_gp.get_alpha_samples_fn()
            compute_exact_posterior_samples_fn = exact_gp.get_posterior_samples_fn(train_ds, test_ds, zero_mean)
            
            alpha_samples_exact = compute_exact_alpha_samples_fn(f0_samples_train, eps0_samples)
            compute_exact_posterior_samples_fn(alpha_samples_exact, f0_samples_test)
            
            # exact_samples = ExactSamplesTuple(
            #     alpha_samples=alpha_samples_exact,
            #     posterior_samples=posterior_samples_exact,
            #     test_rmse=test_rmse_exact_sample,
            #     alpha_map=exact_gp.alpha,
            # )
        
        target_tuples = compute_target_tuples_fn(f0_samples_train, eps0_samples) # (n_samples, TargetTuples)

        alphas, alphas_polyak = jnp.zeros((n_samples, train_ds.N)), jnp.zeros((n_samples, train_ds.N))
        opt_states = optimizer.init(alphas)
        
        for i in tqdm(range(config.iterations)):
            idx_key, step_key, feature_key = jr.split(samples_optim_key, 3)
            features = jax.jit(feature_fn)(key=feature_key)
            idx = idx_fn(jr.split(idx_key, n_samples))
            
            alphas, alphas_polyak, opt_states = update_fn(alphas, alphas_polyak, idx, features, opt_states, target_tuples)
        
        print(f'alphas_polyak: {alphas_polyak.shape}')
        
        posterior_samples = compute_posterior_samples_fn(alphas_polyak, f0_samples_test)  # (n_samples, n_test)
        
        return posterior_samples, alphas_polyak
        
            
            
            
            