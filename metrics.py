
import jax
import jax.random as jr
import jax.numpy as jnp
from chex import Array
from typing import Optional
from utils import revert_z_score


def grad_var_fn(params, grad_fn, num_evals=100):
    grad_var_key = jr.split(jr.PRNGKey(12345), num_evals)
    grad_samples = jax.vmap(grad_fn, (None, 0))(params, grad_var_key)
    grad_var = jnp.var(grad_samples, axis=0).mean()
    
    return grad_var


def RMSE(x: Array, x_hat: Array, mu: Optional[Array]=None, sigma: Optional[Array]=None):
    if mu is not None and sigma is not None:
        x = revert_z_score(x, mu, sigma)
        x_hat = revert_z_score(x_hat, mu, sigma)
    return jnp.sqrt(jnp.mean((x - x_hat) ** 2))




