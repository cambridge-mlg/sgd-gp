from functools import partial
from typing import Callable, Optional

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from chex import Array
from scalable_gps.utils import HparamsTuple, TargetTuple


def i_error(
    params: Array, idx: Array, x: Array, z: Array, target: Array, kernel_fn: Callable
):
    chex.assert_equal_shape_suffix([x, z], suffix_len=1)
    chex.assert_equal_shape_prefix([x, target], prefix_len=1)
    chex.assert_equal_shape_prefix([params, z], prefix_len=1)

    K = kernel_fn(x[idx], z)
    B, N = idx.shape[0], x.shape[0]
    return 0.5 * (N / B) * jnp.sum((target[idx] - K @ params) ** 2)


def i_regularizer(
    params: Array,
    features_x: Array,
    features_z: Array,
    target: Array,
    noise_scale: float,
):
    chex.assert_equal_shape_suffix([features_x, features_z], suffix_len=1)
    chex.assert_equal_shape_prefix([params, features_z], prefix_len=1)
    chex.assert_equal_shape_prefix([target, features_x], prefix_len=1)
    # features_x (num_points, num_features)
    # features_z (num_inducing, num_features)
    params = noise_scale * params
    target = target / noise_scale
    params_norm = ((params @ features_z) ** 2).sum()
    targets_norm = ((target @ features_x) ** 2).sum()
    cross_norm = jnp.dot(params @ features_z, target @ features_x)
    return 0.5 * (params_norm + targets_norm - 2 * cross_norm)


# TODO: pmap over idx / B
def i_error_grad_sample(
    params: Array, idx: Array, x: Array, z: Array, target: Array, kernel_fn: Callable
):
    chex.assert_equal_shape_suffix([x, z], suffix_len=1)
    chex.assert_equal_shape_prefix([x, target], prefix_len=1)
    chex.assert_equal_shape_prefix([params, z], prefix_len=1)

    K = kernel_fn(x[idx], z)
    B, N = K.shape
    return -K.T @ (target[idx] - K @ params) * (N / B)


def i_regularizer_grad_sample(
    params: Array,
    features_x: Array,
    features_z: Array,
    target: Array,
    noise_scale: float,
):
    chex.assert_equal_shape_suffix([features_x, features_z], suffix_len=1)
    chex.assert_equal_shape_prefix([params, features_z], prefix_len=1)
    chex.assert_equal_shape_prefix([target, features_x], prefix_len=1)

    # target_grad = 0 ## features_x @ (features_x.T @ target)
    params = (noise_scale**2) * params
    param_grad = features_z @ (features_z.T @ params)
    cross_grad = features_z @ (features_x.T @ target)
    return param_grad - 2 * cross_grad


def i_loss_fn(
    params: Array,
    idx: Array,
    x: Array,
    z: Array,
    features_x: Array,
    features_z: Array,
    target_tuple: TargetTuple,
    kernel_fn: Callable,
    noise_scale,
):
    err = i_error(params, idx, x, z, target_tuple.error_target, kernel_fn)
    reg = i_regularizer(
        params, features_x, features_z, target_tuple.regularizer_target, noise_scale
    )
    chex.assert_rank([err, reg], 0)

    return err + reg, err, reg
