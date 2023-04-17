from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from chex import Array
from utils import get_gpu_or_cpu_device

Kernel_fn = Callable[[Array, Array], Array]


@partial(jax.jit, device=get_gpu_or_cpu_device())
def solve_K_inv_v(K: Array, v: Array, noise_scale: float):
    """Solves (K + noise_scale^2 I) x = v for x."""
    return jax.scipy.linalg.solve(K + (noise_scale**2) * jnp.identity(v.shape[0]), v, assume_a='pos')


def KvP(x1: Array, x2: Array, v: Array, kernel_fn: Kernel_fn, **kernel_kwargs):
    # TODO: Allocate memory smartly here maybe.

    def _KvP(x1: Array, x2: Array, v: Array, kernel_fn: Kernel_fn, **kernel_kwargs):
        """Calculates K(x_pred, x_train) @ v, with the kernel matrix between x_pred and x_train."""
        return kernel_fn(x1, x2, **kernel_kwargs) @ v

    def _idx_KvP(carry, idx):
        return carry, _KvP(x1[idx], x2, v, kernel_fn, **kernel_kwargs)

    idx_vec = jnp.array([jnp.arange(x1.shape[0])])
    # idx_vec
    return jax.lax.scan(_idx_KvP, jnp.zeros(()), idx_vec)[1].squeeze()


def pivoted_cholesky(matrix, max_rank):

    rank = 0
    diagonal = jnp.diagonal(matrix)
    L = jnp.zeros_like(matrix)

    while rank < max_rank:
        idx = jnp.argmax(diagonal[rank:])
        idx += rank
        L = jax.ops.index_update(L, jax.ops.index[rank, idx], 1)

        temp = jnp.sqrt(diagonal[idx])
        L = jax.ops.index_update(L, jax.ops.index[idx, rank], temp)

        row = matrix[idx, rank + 1:] / L[idx, rank]
        L = jax.ops.index_update(L, jax.ops.index[rank + 1:, idx], row)

        diagonal = jax.ops.index_update(diagonal, idx, 0)
        diagonal = jax.ops.index_update(diagonal, jax.ops.index[rank + 1:], diagonal[rank + 1:] - L[rank + 1:, idx] ** 2)
        rank += 1

    return L[:rank, :]


def pivoted_cholesky_opt(matrix, max_rank):
    matrix.shape[0]
    diagonal = jnp.diagonal(matrix)
    L = jnp.zeros_like(matrix)

    def cond_fun(carry):
        _, rank, diagonal = carry
        return jnp.logical_and(rank < max_rank, jnp.any(diagonal > 1e-6))

    def body_fun(carry):
        L, rank, diagonal = carry
        idx = jnp.argmax(diagonal[rank:])
        idx += rank
        L = jax.ops.index_update(L, jax.ops.index[rank, idx], 1)

        temp = jnp.sqrt(diagonal[idx])
        L = jax.ops.index_update(L, jax.ops.index[idx, rank], temp)

        row = matrix[idx, rank + 1:] / L[idx, rank]
        L = jax.ops.index_update(L, jax.ops.index[rank + 1:, idx], row)

        diagonal = jax.ops.index_update(diagonal, idx, 0)
        diagonal = jax.ops.index_update(diagonal, jax.ops.index[rank + 1:], diagonal[rank + 1:] - L[rank + 1:, idx] ** 2)

        return L, rank + 1, diagonal

    L, _, _ = jax.lax.while_loop(cond_fun, body_fun, (L, 0, diagonal))
    return L[:max_rank, :]