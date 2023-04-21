from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from chex import Array

from scalable_gps.kernels import Kernel
from scalable_gps.utils import get_gpu_or_cpu_device

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

    # idx_vec = jnp.array([jnp.arange(x1.shape[0])])
    idx_vec = jnp.arange(x1.shape[0])
    return jax.lax.scan(_idx_KvP, jnp.zeros(()), idx_vec)[1].squeeze()


def pivoted_cholesky(kernel: Kernel, x: Array, max_rank: int, diag_rtol: float=1e-3, jitter: float=1e-3):
    n = x.shape[0]
    assert max_rank <= n

    orig_error = kernel.get_signal_scale() ** 2 + jitter
    print(f'orig_error: {orig_error}')
    matrix_diag = orig_error * jnp.ones((n,))

    m = 0
    pchol = jnp.zeros((max_rank, n))
    perm = jnp.arange(n)
    
    @partial(jax.jit, static_argnums=(0))
    def _body_fn(m, pchol, perm, matrix_diag):
        maxi = jnp.argmax(matrix_diag[perm[m:]]) + m
        maxval = matrix_diag[perm][maxi]

        perm = perm.at[..., [m, maxi]].set(perm[..., [maxi, m]])

        # TODO: Figure out where jitter gets added, only where row is computed for same index kernel_fn(i, i)
        row = kernel.kernel_fn(x[perm[m]], x[perm[m + 1:]]).squeeze()

        row -= jnp.sum(pchol[:m+1, perm[m + 1:]] * pchol[:m+1, perm[m:m+1]], axis=-2)
        pivot = jnp.sqrt(maxval)
        row /= pivot

        row = jnp.concatenate([pivot[None], row], axis=-1)
        matrix_diag = matrix_diag.at[perm[m:]].set(matrix_diag[perm[m:]] - row**2)

        pchol = pchol.at[m, perm[m:]].set(row)
        
        return pchol, perm, matrix_diag

    cond = True
    while cond:
        pchol, perm, matrix_diag = _body_fn(m, pchol, perm, matrix_diag)
        m = m + 1
        error = jnp.linalg.norm(matrix_diag, ord=1, axis=-1)
        max_err = jnp.max(error / orig_error)
        print(f'Iteration: {m}, error : {max_err}')
        cond = (m < max_rank) and (max_err > diag_rtol)
        
    
    pchol = jnp.swapaxes(pchol, -1, -2)
    return pchol