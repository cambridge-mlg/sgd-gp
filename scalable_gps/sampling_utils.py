from typing import Callable, Tuple

import chex
import jax.numpy as jnp
import jax.random as jr
from chex import Array
from scalable_gps.data import Dataset
from scalable_gps.linalg_utils import KvP


def draw_f0_sample(key: chex.PRNGKey, N: int, L: Array) -> Tuple[Array, Array]:
    """Given L as RFF Features only, computes a sample from the prior function f_0."""
    M = L.shape[-1]
    w_sample = jr.normal(key, (M,))

    f0_sample = L @ w_sample

    f0_sample_train = f0_sample[:N]
    f0_sample_test = f0_sample[N:]

    return f0_sample_train, f0_sample_test, w_sample


def draw_eps0_sample(prior_noise_key, N, noise_scale):
    """Return a sample from the prior noise ε_0."""
    return noise_scale * jr.normal(prior_noise_key, (N,))


def compute_prior_covariance_factor(
    key: chex.PRNGKey,
    train_ds: Dataset,
    test_ds: Dataset,
    feature_params_fn: Callable,
    feature_fn: Callable,
    n_features: int = 0,
):
    """Compute prior_covariance factor L as either chol(K) or Phi Phi^T."""
    x_full = jnp.vstack((train_ds.x, test_ds.x))

    feature_params = feature_params_fn(key, n_features, train_ds.D)
    L = feature_fn(x_full, feature_params)

    return L


def compute_posterior_fn_sample(
    train_ds: Dataset,
    test_ds: Dataset,
    alpha_sample: Array,
    alpha_map: Array,
    f0_sample_test: Array,
    kernel_fn: Callable,
    zero_mean: bool = True,
) -> Array:
    """Compute (~zero_mean) (K(·)x(Kxx + Σ)^{−1} y) + f0(·) − K(·) @ alpha_sample.
    Will use inducing points if available in train_ds."""
    if zero_mean:
        alpha = -alpha_sample
    else:
        alpha = alpha_map - alpha_sample

    inducing_points = train_ds.x if train_ds.z is None else train_ds.z
    posterior_fn_sample = f0_sample_test + KvP(
        test_ds.x, inducing_points, alpha, kernel_fn=kernel_fn
    )
    return posterior_fn_sample
