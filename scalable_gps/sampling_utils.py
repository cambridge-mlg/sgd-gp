from typing import Callable, Tuple

import chex
import jax.numpy as jnp
import jax.random as jr
from chex import Array
from data import Dataset
from linalg_utils import KvP


def draw_f0_sample(
    key: chex.PRNGKey,
    N: int,
    L: Array,
    use_rff: bool = False) -> Tuple[Array, Array]:
    """Given L as either chol(K) or RFF Features, computes a sample from the prior function f_0."""
    if use_rff:
        M = L.shape[-1]
        eps = jr.normal(key, (M,))
    else:
        N_full = L.shape[0]
        eps = jr.normal(key, (N_full,))
    
    f0_sample = L @ eps

    f0_sample_train = f0_sample[:N]
    f0_sample_test = f0_sample[N:]

    return f0_sample_train, f0_sample_test
    

def draw_eps0_sample(prior_noise_key, N, noise_scale):
    """Return a sample from the prior noise ε_0."""
    return noise_scale * jr.normal(prior_noise_key, (N,))


def compute_prior_covariance_factor(
    key: chex.PRNGKey, 
    train_ds: Dataset, 
    test_ds: Dataset, 
    kernel_fn: Callable,
    feature_fn: Callable,
    use_rff: bool = False, 
    n_features: int = 0,
    chol_eps: float = 1e-5):
    """Compute prior_covariance factor L as either chol(K) or Phi Phi^T."""
    x_full = jnp.vstack((train_ds.x, test_ds.x))
    N_full, N = x_full.shape[0], train_ds.N

    if use_rff:
        L = feature_fn(key, n_features, x_full)
    else:
        K_full = kernel_fn(x_full, x_full)
        L = jnp.linalg.cholesky(K_full + chol_eps * jnp.identity(N_full))
    
    return L


def compute_posterior_fn_sample(
    train_ds: Dataset,
    test_ds: Dataset,
    alpha_sample: Array,
    alpha_map: Array,
    f0_sample_test: Array,
    kernel_fn: Callable,
    zero_mean: bool = True
) -> Array:
    """Compute (~zero_mean) (K(·)x(Kxx + Σ)^{−1} y) + f0(·) − K(·) @ alpha_sample."""
    if zero_mean:
        alpha = -alpha_sample
    else:
        alpha = alpha_map - alpha_sample
    posterior_fn_sample = f0_sample_test + KvP(test_ds.x, train_ds.x, alpha, kernel_fn=kernel_fn)
    return posterior_fn_sample

