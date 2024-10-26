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
    return jax.scipy.linalg.solve(
        K + (noise_scale**2) * jnp.identity(v.shape[0]), v, assume_a="pos"
    )


def KvP(
    x1: Array, x2: Array, v: Array, kernel_fn: Kernel_fn, batch_size=1, **kernel_kwargs
):
    """
    Compute the product of the kernel matrix and a vector.

    Args:
        x1 (Array): The first input array of shape (n1, d1).
        x2 (Array): The second input array of shape (n2, d1).
        v (Array): The vector of shape (n2,).
        kernel_fn (Kernel_fn): The kernel function.
        batch_size (int, optional): The batch size for computation. Defaults to 1.
        **kernel_kwargs: Additional keyword arguments to be passed to the kernel function.

    Returns:
        Array: The result of the product of the kernel matrix and the vector, reshaped to (n1,).

    """
    
    def _KvP(_, idx):
        return _, kernel_fn(x1[idx], x2, **kernel_kwargs) @ v

    n1, d1 = x1.shape
    if (n1 % batch_size) > 0:
        padding = batch_size - (n1 % batch_size)
        x1 = jnp.concatenate([x1, jnp.zeros((padding, d1))], axis=0)

    xs = jnp.reshape(jnp.arange(0, x1.shape[0]), (-1, batch_size))
    return jax.lax.scan(_KvP, jnp.zeros(()), xs)[1].reshape(-1)[:n1]


def pivoted_cholesky(
    kernel: Kernel,
    x: Array,
    max_rank: int,
    diag_rtol: float = 1e-3,
    jitter: float = 1e-3,
):
    """
    Performs pivoted Cholesky decomposition on a given kernel matrix.

    Args:
        kernel (Kernel): The kernel object representing the covariance function.
        x (Array): The input data array of shape (n, d), where n is the number of data points and d is the dimensionality.
        max_rank (int): The maximum rank of the Cholesky decomposition.
        diag_rtol (float, optional): The relative tolerance for the maximum error in the diagonal elements. Defaults to 1e-3.
        jitter (float, optional): The jitter value added to the diagonal elements for numerical stability. Defaults to 1e-3.

    Returns:
        Array: The pivoted Cholesky decomposition of the kernel matrix, of shape (max_rank, n).

    Raises:
        AssertionError: If max_rank is greater than the number of data points.

    """
    
    n = x.shape[0]
    assert max_rank <= n

    orig_error = kernel.get_signal_scale() ** 2 + jitter
    print(f"orig_error: {orig_error}")
    matrix_diag = orig_error * jnp.ones((n,))

    m = 0
    pchol = jnp.zeros((max_rank, n))
    perm = jnp.arange(n)

    def _body_fn(m, pchol, perm, matrix_diag):
        maxi = jnp.argmax(matrix_diag[perm[m:]]) + m
        maxval = matrix_diag[perm][maxi]

        perm = perm.at[..., [m, maxi]].set(perm[..., [maxi, m]])

        # TODO: Figure out where jitter gets added, only where row is computed for same index kernel_fn(i, i)
        row = kernel.kernel_fn(x[perm[m]], x[perm[m + 1 :]]).squeeze()

        row -= jnp.sum(
            pchol[: m + 1, perm[m + 1 :]] * pchol[: m + 1, perm[m : m + 1]], axis=-2
        )
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
        print(f"Iteration: {m}, error : {max_err}")
        cond = (m < max_rank) and (max_err > diag_rtol)

    pchol = jnp.swapaxes(pchol, -1, -2)
    return pchol
