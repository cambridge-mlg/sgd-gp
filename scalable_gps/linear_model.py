from functools import partial
from typing import Callable, Optional

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from chex import Array
from scalable_gps.utils import HparamsTuple, TargetTuple


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


def grad_sample(
    params: Array,
    idx: Array,
    x: Array,
    features: Array,
    target_tuple: TargetTuple,
    kernel_fn: Callable,
    noise_scale: float,
):
    K = kernel_fn(x[idx], x)
    B, N = K.shape
    L = features

    err_grad = -K.T @ (target_tuple.error_target[idx] - K @ params) * (N / B)
    reg_grad = L @ (L.T @ ((noise_scale**2) * params - target_tuple.regularizer_target))
    grad = err_grad + reg_grad
    return grad


def improved_grad_sample_batch_kvp(
    params: Array,
    idx: Array,
    x: Array,
    features: Array,
    target_tuple: TargetTuple,
    kernel_fn: Callable,
    noise_scale: float,
):
    K = kernel_fn(x[idx], x)
    B, N = K.shape

    batch_pred = jnp.zeros_like(params)
    batch_pred = batch_pred.at[idx].set(K @ params)

    err_grad = (N / B) * batch_pred - target_tuple.error_target
    reg_grad = (noise_scale**2) * params - target_tuple.regularizer_target
    grad = err_grad + reg_grad
    return grad


def improved_grad_sample_batch_err(
    params: Array,
    idx: Array,
    x: Array,
    features: Array,
    target_tuple: TargetTuple,
    kernel_fn: Callable,
    noise_scale: float,
):
    K = kernel_fn(x[idx], x)
    B, N = K.shape

    batch_err = jnp.zeros_like(params)
    batch_err = batch_err.at[idx].set(K @ params - target_tuple.error_target[idx])

    err_grad = (N / B) * batch_err
    reg_grad = (noise_scale**2) * params - target_tuple.regularizer_target
    grad = err_grad + reg_grad
    return grad


def improved_grad_sample_batch_all(
    params: Array,
    idx: Array,
    x: Array,
    features: Array,
    target_tuple: TargetTuple,
    kernel_fn: Callable,
    noise_scale: float,
):
    
    K = kernel_fn(x[idx], x)
    B, N = K.shape

    batch_err_grad = K @ params - target_tuple.error_target[idx]
    batch_reg_grad = (noise_scale**2) * params[idx] - target_tuple.regularizer_target[
        idx
    ]

    grad = jnp.zeros_like(params)
    return (N / B) * grad.at[idx].set(batch_err_grad + batch_reg_grad)


def improved_grad_sample_random_kvp(
    params: Array,
    idx: Array,
    x: Array,
    features: Array,
    target_tuple: TargetTuple,
    kernel_fn: Callable,
    noise_scale: float,
):
    
    err_grad = features @ (features.T @ params) - target_tuple.error_target
    reg_grad = (noise_scale**2) * params - target_tuple.regularizer_target
    grad = err_grad + reg_grad
    return grad


def loss_fn(
    params: Array,
    idx: Array,
    x: Array,
    features: Array,
    target_tuple: TargetTuple,
    kernel_fn: Callable,
    noise_scale,
):
    """
    Calculates the loss function for the linear model.

    Args:
        params (Array): The model parameters.
        idx (Array): The indices of the data points.
        x (Array): The input data.
        features (Array): The features of the data.
        target_tuple (TargetTuple): The target tuple containing the error target and regularizer target.
        kernel_fn (Callable): The kernel function.
        noise_scale: The scale of the noise.

    Returns:
        float: The calculated loss.

    Raises:
        AssertionError: If the error and regularizer have a rank other than 0.
    """
    
    err = error(params, idx, x, target_tuple.error_target, kernel_fn)
    reg = regularizer(params, features, target_tuple.regularizer_target, noise_scale)
    chex.assert_rank([err, reg], 0)

    loss = err + reg
    return loss


def marginal_likelihood(
    x: Array,
    targets: Array,
    kernel_fn: Callable,
    hparams_tuple: HparamsTuple,
    transform: Optional[Callable] = None,
):
    """
    Calculate the marginal likelihood of a Gaussian process regression model.

    Args:
        x (Array): Input data points.
        targets (Array): Target values.
        kernel_fn (Callable): Kernel function used for computing the covariance matrix.
        hparams_tuple (HparamsTuple): Tuple of hyperparameters for the kernel function.
        transform (Optional[Callable]): Optional transformation function for the hyperparameters.

    Returns:
        float: The marginal likelihood of the Gaussian process regression model.
    """
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

    data_fit_term = -0.5 * jnp.dot(
        targets, jax.scipy.linalg.cho_solve((K_cho_factor, lower), targets)
    )

    log_det_term = -jnp.log(jnp.diag(K_cho_factor)).sum()

    const_term = -(N / 2.0) * jnp.log(2.0 * jnp.pi)

    return data_fit_term + log_det_term + const_term


@partial(jax.jit, backend="cpu")
def exact_solution(targets, K, noise_scale):
    """
    Computes the exact solution of a linear model using the given targets, covariance matrix, and noise scale.

    Args:
        targets: A 1-D array-like object representing the target values.
        K: A 2-D array-like object representing the covariance matrix.
        noise_scale: A scalar value representing the scale of the noise.

    Returns:
        A 1-D array-like object representing the exact solution of the linear model.
    """
    return jax.scipy.linalg.solve(
        K + (noise_scale**2) * jnp.identity(targets.shape[0]), targets, assume_a="pos"
    )
