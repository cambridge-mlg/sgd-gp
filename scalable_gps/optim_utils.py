from typing import Callable

import jax
import jax.numpy as jnp
import optax
from chex import Array
from jax.random import jr
from linear_model import (
    error_grad_sample,
    regularizer_grad_sample,
)
from utils import TargetTuple


def get_stochastic_gradient_fn(
    x: Array,
    kernel_fn: Callable,
    noise_scale: float
):
    def _fn(params, idx, features, target_tuple):
        error_grad = error_grad_sample(params, idx, x, target_tuple.error_target, kernel_fn)
        regularizer_grad = regularizer_grad_sample(
            params,
            features,
            target_tuple.regularizer_target,
            noise_scale,
        )
        return error_grad + regularizer_grad

    return _fn

def get_update_fn(grad_fn: Callable, optimizer, polyak_step_size: float, vmap: bool = False):

    def _fn(params, params_polyak, idx, features, opt_state, target_tuple):
        n_train = params.shape[0]
        grad = grad_fn(params, idx, features, target_tuple) / n_train

        updates, opt_state = optimizer.update(grad, opt_state)
        new_params = optax.apply_updates(params, updates)
        new_params_polyak = optax.incremental_update(
            new_params, params_polyak, step_size=polyak_step_size
        )

        return new_params, new_params_polyak, opt_state

    if vmap:
        return jax.jit(jax.vmap(_fn, in_axes=(0, 0, 0, None, 0, 0)))
    return jax.jit(_fn)


def get_target_tuples_fn(loss_objective: int):

    def _fn(f0_sample_train, eps0_sample):
        n_train = f0_sample_train.shape[0]
        if loss_objective == 1:
            target_tuple = TargetTuple(f0_sample_train + eps0_sample, jnp.zeros((n_train,)))
        elif loss_objective == 2:
            target_tuple = TargetTuple(f0_sample_train, eps0_sample)
        elif loss_objective == 3:
            target_tuple = TargetTuple(jnp.zeros((n_train,)), f0_sample_train + eps0_sample)
        else:
            raise ValueError("loss_type must be 1, 2 or 3")
        
        return target_tuple
    
    return jax.jit(jax.vmap(_fn))


def get_idx_fn(batch_size: int, n_train: int):
    
    def _fn(key):
        idx = jr.randint(key, shape=(batch_size,), minval=0, maxval=n_train)
        
        return idx
    
    return jax.jit(jax.vmap(_fn))