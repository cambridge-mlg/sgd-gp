import jax.numpy as jnp
from chex import Array


def solve_K_inv_v(K: Array, v: Array, noise_scale: float = 1.):
    """Solves (K + noise_scale^2 I) x = v for x."""
    # TODO: use jax.scipy.linalg.solve with sym_pos=True
    # TODO: jit to cpu
    return jnp.linalg.solve(K + (noise_scale ** 2) * jnp.identity(v.shape[0]), v)


def calc_Kstar_v(x_pred, x_train, v, kernel_fn, **kernel_kwargs):
    """Calculates K(x_pred, x_train) @ v, with the kernel matrix between x_pred and x_train."""
    return kernel_fn(x_pred, x_train, **kernel_kwargs) @ v