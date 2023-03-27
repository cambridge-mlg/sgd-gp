from typing import Callable, Optional

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from chex import Array
from utils import revert_z_score


def grad_var_fn(
    params: Array,
    grad_fn: Callable,
    num_evals: int = 100,
    key: chex.PRNGKey = jr.PRNGKey(12345),
):
    grad_var_key = jr.split(key, num_evals)
    grad_samples = jax.vmap(grad_fn, (None, 0))(params, grad_var_key)
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
