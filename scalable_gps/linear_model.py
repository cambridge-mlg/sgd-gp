from functools import partial
from typing import Callable, Optional

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from chex import Array
from utils import HparamsTuple, TargetTuple


def error(params: Array, idx: Array, x: Array, target: Array, kernel_fn: Callable):
    K = kernel_fn(x[idx], x)
    B, N = idx.shape[0], x.shape[0]
    return 0.5 * (N / B) * jnp.sum((target[idx] - K @ params) ** 2)

def regularizer(params: Array, features: Array, target: Array, noise_scale: float):
    params = noise_scale * params
    target = target / noise_scale
    L = features
    R = L.T @ (params - target)
    return 0.5 * jnp.dot(R, R)

# TODO: pmap over idx / B
def error_grad_sample(params: Array, idx: Array, x: Array, target: Array, kernel_fn: Callable):
    K = kernel_fn(x[idx], x)
    B, N = K.shape
    return -K.T @ (target[idx] - K @ params) * (N / B)

def regularizer_grad_sample(params: Array, features: Array, target: Array, noise_scale: float):
    L = features
    params = (noise_scale**2) * params
    return L @ (L.T @ (params - target))


def loss_fn(params: Array, idx: Array, x: Array, features:Array, target_tuple: TargetTuple, kernel_fn: Callable, noise_scale):
    err = error(params, idx, x, target_tuple.error_target, kernel_fn)
    reg = regularizer(params, features, target_tuple.regularizer_target, noise_scale)
    chex.assert_rank([err, reg], 0)
    
    return err + reg, err, reg


def marginal_likelihood(x: Array, targets: Array, kernel_fn: Callable, hparams_tuple: HparamsTuple, 
                        transform: Optional[Callable] = None):
    N = targets.shape[0]
    
    if transform:
        signal_scale = transform(hparams_tuple.signal_scale)
        length_scale = transform(hparams_tuple.length_scale)
        noise_scale = transform(hparams_tuple.noise_scale)
    else:
        signal_scale = hparams_tuple.signal_scale
        length_scale = hparams_tuple.length_scale
        noise_scale = hparams_tuple.noise_scale
    
    K_train = kernel_fn(x, x, signal_scale=signal_scale, length_scale=length_scale)
    K = K_train + (noise_scale**2) * jnp.identity(N)
    K_cho_factor, lower = jax.scipy.linalg.cho_factor(K)
    
    # K_cho_factor = jax.scipy.linalg.cholesky(K)
    # print(K_cho_factor)

    data_fit_term = -0.5 * jnp.dot(
        targets, jax.scipy.linalg.cho_solve((K_cho_factor, lower), targets))
    
    # print(f'data_fit_term: {data_fit_term}')
    log_det_term = -jnp.log(jnp.diag(K_cho_factor)).sum()
    
    # print(f'log_det_term: {log_det_term}')
    const_term = - (N / 2.) * jnp.log(2. * jnp.pi)
    
    # print(f'const_term: {const_term}')

    return data_fit_term + log_det_term + const_term


# def marginal_likelihood(x: Array, targets: Array, kernel_fn: Callable, hparams_tuple: HparamsTuple, 
#                         transform: Optional[Callable] = None):
#     N = targets.shape[0]
    
#     if transform:
#         signal_scale = transform(hparams_tuple.signal_scale)
#         length_scale = transform(hparams_tuple.length_scale)
#         noise_scale = transform(hparams_tuple.noise_scale)
#     else:
#         signal_scale = hparams_tuple.signal_scale
#         length_scale = hparams_tuple.length_scale
#         noise_scale = hparams_tuple.noise_scale
    
#     K_train = kernel_fn(x, x, signal_scale=signal_scale, length_scale=length_scale)
#     K = K_train + (noise_scale**2) * jnp.identity(N)
#     data_fit = -.5 * jnp.dot(targets, jnp.linalg.solve(K, targets))
#     log_det = -.5 * jnp.linalg.slogdet(K)[1]
#     return data_fit + log_det - (N / 2.) * jnp.log(2. * jnp.pi)


@partial(jax.jit, backend='cpu')
def exact_solution(targets, K, noise_scale):
    return jax.scipy.linalg.solve(K + (noise_scale**2) * jnp.identity(targets.shape[0]), targets, assume_a='pos')


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
