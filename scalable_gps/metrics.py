from typing import Callable, Optional

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from chex import Array
from data import Dataset
from utils import revert_z_score


def grad_var_fn(
    params: Array,
    grad_fn: Callable,
    batch_size: int,
    train_ds: Dataset,
    feature_fn: Callable, 
    num_evals: int = 100,
    key: chex.PRNGKey = jr.PRNGKey(12345),
):
    B, N = batch_size, train_ds.N
    @jax.jit
    def _compute_grad(single_key):
        idx_key, feature_key = jr.split(single_key)
        idx = jr.randint(idx_key, shape=(B,), minval=0, maxval=N)
        features = feature_fn(key=feature_key, x=train_ds.x, recompute=False)
        grad_val = grad_fn(params, idx, features)

        return grad_val
    grad_var_key = jr.split(key, num_evals)
    grad_samples = jax.vmap(_compute_grad)(grad_var_key)
    grad_var = jnp.var(grad_samples, axis=0).mean()

    return grad_var


def hilbert_space_RMSE(x: Array, x_hat: Array, K: Array):
    return jnp.sqrt(jnp.mean((x - x_hat) * (K @ (x - x_hat))))

def RMSE(
    x: Array, x_hat: Array, mu: Optional[Array] = None, sigma: Optional[Array] = None
):
    if mu is not None and sigma is not None:
        x = revert_z_score(x, mu, sigma)
        x_hat = revert_z_score(x_hat, mu, sigma)
    return jnp.sqrt(jnp.mean((x - x_hat) ** 2))
