from functools import partial
from typing import List, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import ml_collections
from chex import Array
from data import Dataset
from kernels import Kernel
from linalg_utils import KvP, solve_K_inv_v
from linear_model import (
    draw_prior_noise_sample,
    loss_fn,
)
from metrics import RMSE
from train_utils import (
    get_eval_fn,
    get_stochastic_gradient_fn,
    get_update_fn,
    train,
)
from utils import ExactMetricsTuple, ExactSamplesTuple, TargetTuple


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

    def calculate_test_rmse(self, train_ds: Dataset, test_ds: Dataset) -> Tuple[Array, Array]:
        y_pred = self.predictive_mean(train_ds, test_ds)
        test_rmse = RMSE(y_pred, test_ds.y, mu=train_ds.mu_y, sigma=train_ds.sigma_y)

        return test_rmse, y_pred

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

    def compute_prior_fn_sample(
        self,
        key: chex.PRNGKey,
        N: int,
        L: Array,
        use_rff: bool = False,
    ) -> Tuple[Array, Array]:
        """Given L as either chol(K) or RFF Features, computes a sample from the prior function f_0."""
        if use_rff:
            M = L.shape[-1]
            eps = jr.normal(key, (M,))
        else:
            N_full = L.shape[0]
            eps = jr.normal(key, (N_full,))
        
        prior_fn_sample = L @ eps

        prior_fn_sample_train = prior_fn_sample[:N]
        prior_fn_sample_test = prior_fn_sample[N:]

        return prior_fn_sample_train, prior_fn_sample_test
    
    def compute_posterior_fn_sample(
        self,
        train_ds: Dataset,
        test_ds: Dataset,
        alpha_sample: Array,
        prior_fn_sample_test: Array,
        zero_mean: bool = True
    ) -> Array:
        """Compute (~zero_mean) (K(·)x(Kxx + Σ)^{−1} y) + f0(·) − K(·) (Kxx + Σ)^{−1} (f0(x) + ε0)."""
        if zero_mean:
            alpha = -alpha_sample
        else:
            alpha = self.alpha - alpha_sample
        posterior_fn_sample = prior_fn_sample_test + KvP(test_ds.x, train_ds.x, alpha, kernel_fn=self.kernel.kernel_fn)
        return posterior_fn_sample
    
    def compute_representer_weights_sample(
        self,
        key: chex.PRNGKey,
        train_ds: Dataset,
        prior_fn_sample_train: Array
    ) -> Array:
        """Computes (Kxx+ Σ)^{-1} (f0(x) + ε0)"""
        N = train_ds.N
        # draw prior noise sample ε0
        prior_noise_sample = draw_prior_noise_sample(key, N, noise_scale=self.noise_scale)

        alpha_sample = solve_K_inv_v(
            self.K,
            prior_fn_sample_train + prior_noise_sample,
            noise_scale=self.noise_scale,
        )

        return alpha_sample
    
    def compute_L(
        self, 
        key: chex.PRNGKey, 
        train_ds: Dataset, 
        test_ds: Dataset, 
        use_rff: bool = False, 
        n_features: int = 0,
        chol_eps: float = 1e-6):
        """Compute L as either chol(K) or RFF Features."""
        x_full = jnp.vstack((train_ds.x, test_ds.x))
        N_full, N = x_full.shape[0], train_ds.N

        if use_rff:
            L = self.kernel.feature_fn(key, n_features, x_full)
            K_train = L[:N] @ L[:N].T
        else:
            K_full = self.kernel.kernel_fn(x_full, x_full)
            L = jnp.linalg.cholesky(K_full + chol_eps * jnp.identity(N_full))
            K_train = K_full[:N, :N]
        
        return L, K_train


    def compute_zero_mean_samples(
        self,
        key: chex.PRNGKey,
        n_samples: int,
        train_ds: Dataset,
        test_ds: Dataset,
        n_features: int = 0,
        use_rff: bool = True,
        chol_eps: float = 1e-5,
        L: Optional[Array] = None,
    ):  
        """Vmapped utility fn that computes n_samples with ExactGP. Can optionally pass in L, either chol(K) or RFF features."""
        key, prior_features_sample_key = jr.split(key)
        if L is None:
            L, K_train = self.compute_L(prior_features_sample_key, train_ds, test_ds, use_rff, n_features, chol_eps)
            self.K = K_train if self.K is None else self.K  # TODO: Do we need to do this?

        @jax.jit
        def _compute_fn(single_key):
            prior_key, sample_opt_key = jr.split(single_key)
            prior_fn_sample_train, prior_fn_sample_test = self.compute_prior_fn_sample(
                    prior_key, train_ds.N, L, use_rff=use_rff)

            alpha_sample = self.compute_representer_weights_sample(
                sample_opt_key,
                train_ds,
                prior_fn_sample_train)
            
            zero_mean_posterior_sample = self.compute_posterior_fn_sample(
                train_ds, test_ds, alpha_sample, prior_fn_sample_test, zero_mean=True)
        
            return zero_mean_posterior_sample, alpha_sample
        
        keys = jr.split(key, n_samples)
        
        return jax.vmap(jax.jit(_compute_fn))(keys)  # (n_samples, n_test), (n_samples, n_train)


class SGDGPModel(ExactGPModel):
    def __init__(self, noise_scale: float, kernel: Kernel, **kwargs):
        super().__init__(noise_scale=noise_scale, kernel=kernel, **kwargs)

    def _init_params(self, train_ds: Dataset) -> Tuple[Array, Array]:
        # TODO: consider different init for sampling.
        alpha = jnp.zeros((train_ds.N,))
        alpha_polyak = jnp.zeros((train_ds.N,))   
        
        return alpha, alpha_polyak

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

    def compute_representer_weights_sample(
        self,
        key: chex.PRNGKey,
        train_ds: Dataset,
        test_ds: Dataset,
        prior_fn_sample_train: Array,
        prior_fn_sample_test: Array,
        config: ml_collections.ConfigDict,
        loss_type: int,
        metrics: List[str],
        metrics_prefix: str = "",
        exact_samples: Optional[ExactSamplesTuple] = None
    ):
        """Compute the representer weights alpha_sample by solving (Kxx+ Σ)^{-1} (f0(x) + ε0) using SGD."""
        # draw prior noise sample ε0
        prior_noise_sample = draw_prior_noise_sample(key, train_ds.N, noise_scale=self.noise_scale)

        # Depending on the three types of losses we can compute the gradient of the loss function accordingly
        if loss_type == 1:
            target_tuple = TargetTuple(prior_fn_sample_train + prior_noise_sample, jnp.zeros_like(train_ds.y))
        elif loss_type == 2:
            target_tuple = TargetTuple(prior_fn_sample_train, prior_noise_sample)
        elif loss_type == 3:
            target_tuple = TargetTuple(jnp.zeros_like(train_ds.y), prior_fn_sample_train + prior_noise_sample)
        else:
            raise ValueError("loss_type must be 1, 2 or 3")

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
            n_features=config.n_features,
            recompute=config.recompute_features)

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
            exact_samples=exact_samples,
        )

        # Initialise alpha and alpha_polyak for sample
        init_alpha, init_alpha_polyak = self._init_params(train_ds)

        alpha_polyak, aux = train(
            key, config, update_fn, eval_fn, feature_fn, train_ds, init_alpha, init_alpha_polyak
        )

        return alpha_polyak, aux
    
    def compute_zero_mean_samples(
        self,
        key: chex.PRNGKey,
        n_samples: int,
        train_ds: Dataset,
        test_ds: Dataset,
        config: ml_collections.ConfigDict,
        metrics: List[str],
        use_rff: bool = True,
        chol_eps: float = 1e-5,
        compare_exact: bool = False
    ):
        """Vmapped utility fn that computes n_samples using SGD. Optionally compares to ExactGP."""
        # compute shared RFF
        key, prior_features_sample_key = jr.split(key)
        L, K_train = self.compute_L(
            prior_features_sample_key, train_ds, test_ds, use_rff, config.n_features_prior_sample, chol_eps)

        # compare to ExactGP
        if compare_exact:
            exact_gp = ExactGPModel(self.noise_scale, self.kernel)
            exact_gp.K = K_train
            exact_gp.compute_representer_weights(train_ds)

        # call vmapped SGDGP, pass in L (either chol(K) or RFF features that prior samples are drawn from)
        def _compute_fn(single_key):
            vmap_idx = jax.lax.axis_index('sample')
            prior_sample_key, sample_key = jr.split(single_key)

            # draw a prior function sample
            prior_fn_sample_train, prior_fn_sample_test = self.compute_prior_fn_sample(
                    prior_sample_key, train_ds.N, L, use_rff=use_rff)

            if compare_exact:
                exact_alpha_sample = exact_gp.compute_representer_weights_sample(sample_key, train_ds, prior_fn_sample_train)
                # We compute K(·) (Kxx + Σ)−1 y − K(·) (Kxx + Σ)^{−1} (f0(x) + ε0), ignoring the f0(.) term in Eq. (4).
                y_pred_exact_sample = KvP(test_ds.x, train_ds.x, exact_gp.alpha - exact_alpha_sample, kernel_fn=exact_gp.kernel.kernel_fn)
                test_rmse_exact_sample = RMSE(test_ds.y, y_pred_exact_sample, mu=train_ds.mu_y, sigma=train_ds.sigma_y)
                
                exact_samples = ExactSamplesTuple(
                    alpha=exact_alpha_sample,
                    y_pred=y_pred_exact_sample,
                    test_rmse=test_rmse_exact_sample,
                    alpha_map=exact_gp.alpha,
                )
            else:
                exact_samples = None

            # Compute a posterior sample
            alpha_sample, info = self.compute_representer_weights_sample(
                sample_key,
                train_ds,
                test_ds,
                prior_fn_sample_train,
                prior_fn_sample_test,
                config,
                config.loss_objective,
                metrics,
                metrics_prefix=f"sampling/{vmap_idx}",
                exact_samples=exact_samples,
            )
            zero_mean_posterior_sample = self.compute_posterior_fn_sample(
                train_ds, test_ds, alpha_sample, prior_fn_sample_test, zero_mean=True)

            return zero_mean_posterior_sample, alpha_sample
        
        keys = jr.split(key, n_samples)
        
        # TODO: jitting this function causes an indefinite hang after optimisation is complete.
        return jax.vmap(_compute_fn, axis_name='sample')(keys)  # (n_samples, n_test), (n_samples, n_train)


