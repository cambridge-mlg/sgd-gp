from typing import List, Optional

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
    get_sampling_eval_fn,
    get_stochastic_gradient_fn,
    get_update_fn,
    train,
)
from utils import ExactValsTuple, TargetTuple


class Model:
    def __init__(self):
        pass

    def compute_representer_weights(self):
        raise NotImplementedError(
            "compute_representer_weights() must be implemented in the derived class."
        )

    def compute_posterior_sample(self):
        raise NotImplementedError(
            "compute_posterior_sample() must be implemented in the derived class."
        )


class ExactGPModel(Model):
    def __init__(self, noise_scale: float, kernel: Kernel):
        super().__init__()

        self.noise_scale = noise_scale
        self.kernel = kernel
        self.K = None
        self.alpha = None

    def set_K(self, train_x: Array, recompute: bool = False):
        """Compute the kernel matrix K if it is None or recompute is True."""
        if self.K is None or recompute:
            self.K = self.kernel.K(train_x, train_x)

    def set_alpha(self, train_ds: Dataset, recompute: bool = False):
        """Compute the representer weights alpha if it is None or recompute is True."""
        if self.alpha is None or recompute:
            self.alpha = self.compute_representer_weights(train_ds)

    def compute_representer_weights(self, train_ds: Dataset):
        """Compute the representer weights alpha by solving alpha = (K + sigma^2 I)^{-1} y"""

        # Compute Kernel exactly
        self.set_K(train_ds.x)

        # Compute the representer weights by solving alpha = (K + sigma^2 I)^{-1} y
        self.alpha = solve_K_inv_v(
            self.K, train_ds.y, noise_scale=self.noise_scale
        ).squeeze()

        return self.alpha

    def predictive_mean(self, train_ds: Dataset, test_ds: Dataset):
        # Compute the representer weights by solving alpha = (K + sigma^2 I)^{-1} y

        y_pred = KvP(test_ds.x, train_ds.x, self.alpha, kernel_fn=self.kernel.K)

        return y_pred  # (N_test, 1)

    def predictive_variance(self, train_ds: Dataset, test_ds: Dataset, return_marginal_variance: bool = True):
        """Compute the posterior variance of the test points."""
        K_test = self.kernel.K(test_ds.x, test_ds.x)  # N_test, N_test
        
        K_train_test = self.kernel.K(train_ds.x, test_ds.x)  # N_train, N_test
        
        K_inv_K_train_test = solve_K_inv_v(self.K, K_train_test, noise_scale=self.noise_scale)
    
        variance = K_test - K_train_test.T @ K_inv_K_train_test
        
        if return_marginal_variance:
            return jnp.diag(variance)  # (N_test, 1)
        return variance  # (N_test, N_test)

    def predictive_variance_samples(
        self,
        zero_mean_posterior_samples: Array,
        return_marginal_variance: bool = True):
        # posterior_samples = (N_samples, N_test)
        
        if return_marginal_variance:
            variance = jnp.mean(zero_mean_posterior_samples ** 2, axis=0)  # (N_test, 1)
        else:
            n_samples = zero_mean_posterior_samples.shape[0]
            variance = zero_mean_posterior_samples.T @ zero_mean_posterior_samples / n_samples
    
        return variance

    def calculate_test_rmse(self, train_ds: Dataset, test_ds: Dataset):

        y_pred = self.predictive_mean(train_ds, test_ds)
        test_rmse = RMSE(y_pred, test_ds.y, mu=train_ds.mu_y, sigma=train_ds.sigma_y)

        return test_rmse, y_pred

    def compute_prior_fn_sample(
        self,
        key: chex.PRNGKey,
        train_ds: Dataset,
        test_ds: Dataset,
        n_features: int,
        use_rff: bool = False,
        chol_eps: float = 1e-6,
    ):
        x_full = jnp.vstack((train_ds.x, test_ds.x))
        N_full, N = x_full.shape[0], train_ds.N
        M = n_features

        prior_fn_key, feature_key = jr.split(key, 2)

        if use_rff:
            L = self.kernel.Phi(feature_key, M, x_full)
            eps = jr.normal(prior_fn_key, (M,))
            K_train = L[:N] @ L[:N].T
        else:
            K_full = self.kernel.K(x_full, x_full)
            L = jnp.linalg.cholesky(K_full + chol_eps * jnp.identity(N_full))
            eps = jr.normal(prior_fn_key, (N_full,))
            K_train = K_full[:N, :N]
        
        prior_fn_sample = L @ eps

        prior_fn_sample_train = prior_fn_sample[:N]
        prior_fn_sample_test = prior_fn_sample[N:]

        return prior_fn_sample_train, prior_fn_sample_test, K_train
    
    def compute_zero_mean_posterior_fn_sample(
        self,
        train_ds: Dataset,
        test_ds: Dataset,
        alpha_sample: Array,
        prior_fn_sample_test: Array
    ):
        zero_mean_posterior_fn_sample = prior_fn_sample_test - KvP(test_ds.x, train_ds.x, alpha_sample, kernel_fn=self.kernel.K)
        return zero_mean_posterior_fn_sample    
    
    def compute_representer_weights_sample(
        self,
        key: chex.PRNGKey,
        train_ds: Dataset,
        prior_fn_sample_train: Array,
        K_train: Array,
    ):
        """Computes $(K_{xx}+ \Sigma)^{-1}(f_0(x)+ \varepsilon_0)$"""
        
        N = train_ds.N
        # draw prior noise sample
        prior_noise_sample = draw_prior_noise_sample(
            key, N, noise_scale=self.noise_scale
        )

        alpha_sample = solve_K_inv_v(
            K_train,
            prior_fn_sample_train + prior_noise_sample,
            noise_scale=self.noise_scale,
        )

        return alpha_sample
    
    
    def compute_zero_mean_samples(
        self,
        key: chex.PRNGKey,
        n_samples: int,
        train_ds: Dataset,
        test_ds: Dataset,
        n_features: int = 0,
        use_rff: bool = True,
    ):
        @jax.jit
        def _compute_fn(single_key):
            prior_key, sample_opt_key = jr.split(single_key)
            prior_fn_sample_train, prior_fn_sample_test, K_train = self.compute_prior_fn_sample(
                    prior_key, train_ds, test_ds, n_features, use_rff=use_rff)

            alpha_sample = self.compute_representer_weights_sample(
                sample_opt_key,
                train_ds,
                prior_fn_sample_train,
                K_train)
            
            zero_mean_posterior_sample = self.compute_zero_mean_posterior_fn_sample(
                train_ds, test_ds, alpha_sample, prior_fn_sample_test)
        
            return zero_mean_posterior_sample, alpha_sample
        
        keys = jr.split(key, n_samples)
        
        return jax.vmap(_compute_fn)(keys)  # (n_samples, n_test), (n_samples, n_train)


class SGDGPModel(ExactGPModel):
    def __init__(self, noise_scale: float, kernel: Kernel, **kwargs):
        super().__init__(noise_scale=noise_scale, kernel=kernel, **kwargs)

        # Initialize alpha and alpha_polyak
        self.alpha = None

        self.omega = None
        self.phi = None

    def _init_params(self, train_ds):
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
        compare_exact_vals: Optional[ExactValsTuple] = None,
    ):
        # @ K (alpha + target)
        target_tuple = TargetTuple(
            train_ds.y.squeeze(), jnp.zeros_like(train_ds.y).squeeze()
        )
        grad_fn = get_stochastic_gradient_fn(
            x=train_ds.x,
            target_tuple=target_tuple,
            kernel_fn=self.kernel.K,
            feature_fn=self.kernel.Phi,
            batch_size=config.batch_size,
            num_features=config.num_features,
            noise_scale=self.noise_scale,
            recompute_features=config.recompute_features,
        )

        # Define the gradient update function
        update_fn = get_update_fn(
            grad_fn=grad_fn, 
            n_train=train_ds.N, 
            polyak_step_size=config.polyak)

        eval_fn = get_eval_fn(
            metrics,
            train_ds,
            test_ds,
            loss_fn,
            grad_fn,
            target_tuple,
            kernel_fn=self.kernel.K,
            noise_scale=self.noise_scale,
            metrics_prefix=metrics_prefix,
            compare_exact_vals=compare_exact_vals,
        )

        # Initialise alpha and alpha_polyak
        init_alpha, init_alpha_polyak = self._init_params(train_ds)

        # Solve optimisation problem to obtain representer_weights
        self.alpha, aux = train(
            key, config, update_fn, eval_fn, init_alpha, init_alpha_polyak
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
        compare_exact_vals: Optional[ExactValsTuple] = None,
    ):
        # draw prior noise sample
        prior_noise_sample = draw_prior_noise_sample(
            key, train_ds.N, noise_scale=self.noise_scale
        )

        # Depending on the three types of losses we can compute the gradient of the loss function accordingly
        if loss_type == 1:
            target_tuple = TargetTuple(
                prior_fn_sample_train + prior_noise_sample, jnp.zeros_like(train_ds.y).squeeze()
            )
        elif loss_type == 2:
            target_tuple = TargetTuple(prior_fn_sample_train, prior_noise_sample)
        elif loss_type == 3:
            target_tuple = TargetTuple(
                jnp.zeros_like(train_ds.y).squeeze(), prior_fn_sample_train + prior_noise_sample
            )
        else:
            raise ValueError("loss_type must be 1, 2 or 3")

        grad_fn = get_stochastic_gradient_fn(
            x=train_ds.x,
            target_tuple=target_tuple,
            kernel_fn=self.kernel.K,
            feature_fn=self.kernel.Phi,
            batch_size=config.batch_size,
            num_features=config.num_features_optim,
            noise_scale=self.noise_scale,
            recompute_features=config.recompute_features,
        )

        # Define the gradient update function
        update_fn = get_update_fn(
            grad_fn=grad_fn, 
            n_train=train_ds.N, 
            polyak_step_size=config.polyak)

        eval_fn = get_sampling_eval_fn(
            metrics,
            train_ds,
            test_ds,
            prior_fn_sample_test,
            self.alpha,
            loss_fn,
            grad_fn,
            target_tuple,
            self.kernel.K,
            self.noise_scale,
            metrics_prefix,
            compare_exact_vals,
        )
        eval_fn = None

        # Initialise alpha and alpha_polyak for sample
        init_alpha, init_alpha_polyak = self._init_params(train_ds)

        alpha_polyak, aux = train(
            key, config, update_fn, eval_fn, init_alpha, init_alpha_polyak
        )

        return alpha_polyak, aux
    
    def compute_zero_mean_samples(
        self,
        key: chex.PRNGKey,
        n_samples: int,
        train_ds: Dataset,
        test_ds: Dataset,
        config: ml_collections.ConfigDict,
        sampling_metrics: List[str],
        use_rff: bool = True,
        compare_exact_vals: Optional[ExactValsTuple] = None,
    ):
        def _compute_fn(single_key):
            prior_key, sample_opt_key = jr.split(single_key)
            # Compute a prior sample
            prior_fn_sample_train, prior_fn_sample_test, _ = self.compute_prior_fn_sample(
                    prior_key, train_ds, test_ds, config.num_features_prior_sample, use_rff=use_rff)

            # Compute a posterior sample
            loss_objective = config.loss_objective
            alpha_sample, info = self.compute_representer_weights_sample(
                sample_opt_key,
                train_ds,
                test_ds,
                prior_fn_sample_train,
                prior_fn_sample_test,
                config,
                loss_objective,
                sampling_metrics,
                metrics_prefix=f"sampling_{loss_objective}",
                compare_exact_vals=compare_exact_vals,
            )
            zero_mean_posterior_sample = self.compute_zero_mean_posterior_fn_sample(
                train_ds, test_ds, alpha_sample, prior_fn_sample_test)

            return zero_mean_posterior_sample, alpha_sample
        
        keys = jr.split(key, n_samples)
        
        return jax.vmap(_compute_fn)(keys)  # (n_samples, n_test), (n_samples, n_train)


