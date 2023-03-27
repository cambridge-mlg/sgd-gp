from typing import Callable, Optional

import chex
import jax.numpy as jnp
import jax.random as jr
from chex import Array
from utils import TargetTuple


def error(params, targets, K):
    # TODO: Make this not always take in instantiated kernel, but RFF features etc.
    return 0.5 * jnp.sum((targets - K @ params) ** 2)


# TODO: pmap over B.
# TODO: pass idx directly, can be different sizes depending on pmap and vmap.
def error_grad_sample(params, key, B, x, target, kernel_fn):

    N = x.shape[0]
    idx = jr.randint(key, shape=(B,), minval=0, maxval=N)
    K = kernel_fn(x[idx], x)
    datapoint_grads = -K.T @ (target.squeeze()[idx] - K @ params) * (N / B)
    return datapoint_grads


def regularizer(params, target, K, noise_scale):
    params = (noise_scale**2) * params
    target = target.squeeze()
    return 0.5 * (params - target).T @ K @ (params - target)


def regularizer_grad_sample(
    params, key, M, x, target, feature_fn, noise_scale, recompute_features=True
):
    R = feature_fn(key, M, x, recompute=recompute_features)
    params = (noise_scale**2) * params
    return R @ (R.T @ (params - target.squeeze()))


def loss_fn(params: Array, target_tuple: TargetTuple, K: Array, noise_scale):
    # TODO: MAke the loss function not always take in instantiated kernel, but RFF features etc.
    return (
        error(params, target_tuple.error_target, K).squeeze()
        + regularizer(params, target_tuple.regularizer_target, K, noise_scale).squeeze()
    )


def exact_solution(targets, K, noise_scale):
    # TODO: Add jax.scipy.linalg.solve with pos_sym=True
    return jnp.linalg.solve(
        K + (noise_scale**2) * jnp.identity(targets.shape[0]), targets
    )


def predict(params, x_pred, x_train, kernel_fn, **kernel_kwargs):
    return kernel_fn(x_pred, x_train, **kernel_kwargs) @ params


def draw_prior_function_sample(
    feature_key: chex.PRNGKey,
    prior_function_key: chex.PRNGKey,
    M: int,
    x: Array,
    feature_fn: Callable,
    K: Optional[Array] = None,
    use_chol: bool = False,
    chol_eps: float = 1e-5,
):
    """Returns prior fn sample along with either cholesky factorization of K or feature matrix R."""
    N = x.shape[0]
    if use_chol:
        L = jnp.linalg.cholesky(K + chol_eps * jnp.identity(N))
        eps = jr.normal(prior_function_key, (N,))
    else:
        L = feature_fn(feature_key, M, x)
        eps = jr.normal(prior_function_key, (M,))

    return L @ eps, L


def draw_prior_noise_sample(prior_noise_key, N, noise_scale):
    return noise_scale * jr.normal(prior_noise_key, (N,))
