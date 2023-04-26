import jax
import jax.numpy as jnp
import jax.random as jr
from chex import Array
from typing import Callable

from scalable_gps.kernels import Kernel
from scalable_gps.data import Dataset
from scalable_gps.linalg_utils import KvP


def init(
        seed: int,
        D: int,
        kernel: Kernel, 
        n_features: int = 10000,
        n_data_init: int = 10000,
        minval: float = -1.0,
        maxval: float = 1.0):
    """
    Initialise thompson sampling in unit hypercube [-1, 1]^D
    by constructing the kernel and sampling a function from the GP prior
    """
    omega_key, phi_key, w_key, data_key = jr.split(jr.PRNGKey(seed), 4)

    omega = kernel.omega_fn(omega_key, D, n_features)
    phi = kernel.phi_fn(phi_key, n_features)
    
    w = jr.normal(w_key, shape=(n_features,))

    signal_scale = kernel.get_signal_scale()
    length_scale = kernel.get_length_scale()

    @jax.jit
    def feature_fn(x):
        L = signal_scale * jnp.sqrt(2.0 / n_features) * jnp.cos((x / length_scale) @ omega + phi)
        return L

    @jax.jit
    def objective_fn(x):
        L = feature_fn(x)
        return L @ w

    x_init = jr.uniform(data_key, shape=(n_data_init, D), minval=minval, maxval=maxval)
    y_init = objective_fn(x_init)

    ds_init = Dataset(x_init, y_init, n_data_init, D)
    return feature_fn, objective_fn, ds_init


def get_maximum(ds: Dataset):
    idx = jnp.argmax(ds.y)
    return ds.x[idx], ds.y[idx]


def add_batch(ds: Dataset, x_batch: Array, objective_fn: Callable):
    y_batch = objective_fn(x_batch)
    x = jnp.concatenate([ds.x, x_batch], axis=0)
    y = jnp.concatenate([ds.y, y_batch], axis=0)
    N = ds.N + x_batch.shape[0]
    return Dataset(x, y, N, ds.D)


def get_posterior_sample_fn(w, alpha_sample, feature_fn, kernel_fn, ds, alpha_map, batch_size=1):

    @jax.jit
    def f(x):
        L = feature_fn(x)
        return L @ w + KvP(x, ds.x, alpha_map - alpha_sample, kernel_fn=kernel_fn, batch_size=batch_size)

    return f
    
