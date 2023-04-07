from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from chex import Array
from utils import get_gpu_or_cpu_device

Kernel_fn = Callable[[Array, Array], Array]

# TODO: Check if GPU exists then use GPU
@partial(jax.jit, device=get_gpu_or_cpu_device())
def solve_K_inv_v(K: Array, v: Array, noise_scale: float):
    """Solves (K + noise_scale^2 I) x = v for x."""
    return jax.scipy.linalg.solve(K + (noise_scale**2) * jnp.identity(v.shape[0]), v, assume_a='pos')


@partial(jax.jit, static_argnums=(3,))
def KvP(x1: Array, x2: Array, v: Array, kernel_fn: Kernel_fn, **kernel_kwargs):
    """Calculates K(x_pred, x_train) @ v, with the kernel matrix between x_pred and x_train."""
    # TODO: Minibatch over x_pred potentially, to prevent memory blow up.
    return kernel_fn(x1, x2, **kernel_kwargs) @ v


def batched_KvP(x1: Array, x2: Array, v: Array, kernel_fn: Kernel_fn, **kernel_kwargs):
    # TODO: Allocate memory smartly here maybe.
    @jax.jit
    def idx_KvP(carry, idx):
        return carry, KvP(x1[idx], x2, v, kernel_fn, **kernel_kwargs)

    idx_vec = jnp.arange(x1.shape[0])
    return jax.lax.scan(idx_KvP, jnp.zeros(()), idx_vec)[1].squeeze()
