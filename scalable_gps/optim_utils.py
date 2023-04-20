from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import ml_collections
import optax
from chex import Array
from scalable_gps.linear_model import (
    error_grad_sample,
    regularizer_grad_sample,
)
from scalable_gps.utils import TargetTuple

PyTree = Any

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

    return jax.jit(_fn)


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
        return jax.jit(jax.vmap(_fn, in_axes=(0, 0, None, None, 0, 0)))
    return jax.jit(_fn)


def get_target_tuples_fn(loss_objective: int):

    def _fn(f0_sample_train, eps0_sample):
        n_train = f0_sample_train.shape[0]
        if loss_objective == 1:
            target_tuple = TargetTuple(error_target=f0_sample_train + eps0_sample, regularizer_target=jnp.zeros((n_train,)))
        elif loss_objective == 2:
            target_tuple = TargetTuple(error_target=f0_sample_train, regularizer_target=eps0_sample)
        elif loss_objective == 3:
            target_tuple = TargetTuple(error_target=jnp.zeros((n_train,)), regularizer_target=f0_sample_train + eps0_sample)
        else:
            raise ValueError("loss_type must be 1, 2 or 3")
        
        return target_tuple
    
    return jax.jit(jax.vmap(_fn))


# TODO: rename vmap to share_idx
def get_uniform_idx_fn(batch_size: int, n_train: int, vmap: bool = False):
    
    def _fn(_, key):
        idx = jr.randint(key, shape=(batch_size,), minval=0, maxval=n_train)
        
        return idx

    if vmap:
        return jax.jit(jax.vmap(_fn))
    return jax.jit(_fn)


def get_iterative_idx_fn(batch_size: int, n_train: int):
    
    def _fn(iter, _):
        idx = jnp.arange(batch_size) + ((iter * batch_size) % n_train)
        
        return idx

    return jax.jit(_fn)


def get_idx_fn(batch_size: int, n_train: int, iterative_idx: bool = True, vmap: bool = False):
    if iterative_idx:
        idx_fn = get_iterative_idx_fn(batch_size, n_train)
    else:
        idx_fn = get_uniform_idx_fn(batch_size, n_train, vmap=vmap)
    return idx_fn


# Copied from https://github.com/shreyaspadhy/jaxutils/blob/29781a1ad835653e6709065d77eb1e90a8f60e1a/train/utils.py#L184
def get_lr_and_schedule(
    optim_name: str,
    optim_config: ml_collections.ConfigDict,
    lr_schedule_name: Optional[str],
    lr_schedule_config: Optional[ml_collections.ConfigDict],
    steps_per_epoch: Optional[int] = None,
    model_mask: Optional[PyTree] = None,
):
    """Returns an optimizer with (optional lr_schedule)."""
    if lr_schedule_name is not None and lr_schedule_config is not None:
        schedule = getattr(optax, lr_schedule_name)
        if lr_schedule_name == "piecewise_constant_schedule":
            # Check required configs are present
            required_configs = ["scales_per_epoch"]
            if not all(name in lr_schedule_config for name in required_configs):
                print(lr_schedule_config)
                raise ValueError(f"{lr_schedule_name} requires {required_configs}")

            # Convert scales_per_epoch from str to int, scale by train_loader
            lr_boundaries_and_scales = {}
            scales_per_epoch = lr_schedule_config.get("scales_per_epoch", None)
            for k, v in scales_per_epoch.items():
                boundary = int(k) * steps_per_epoch
                lr_boundaries_and_scales[boundary] = v

            lr_schedule_config.boundaries_and_scales = {
                str(k): v for k, v in lr_boundaries_and_scales.items()
            }

            # Define LR Schedule
            lr = schedule(
                init_value=optim_config.learning_rate,
                boundaries_and_scales=lr_boundaries_and_scales,
            )
        elif lr_schedule_name == "exponential_decay":
            # Check required configs are present
            required_configs = ["decay_rate", "transition_steps"]
            if not all(name in lr_schedule_config for name in required_configs):
                raise ValueError(f"{lr_schedule_name} requires {required_configs}")

            # Define LR Schedule
            lr = schedule(
                init_value=optim_config.learning_rate,
                decay_rate=lr_schedule_config.decay_rate,
                transition_steps=lr_schedule_config.transition_steps,
            )

        elif lr_schedule_name == "warmup_exponential_decay_schedule":
            # Check required configs are present
            required_configs = ["init_value", "warmup_steps", "transition_steps",
                                "decay_rate", "transition_begin",]
            if not all(name in lr_schedule_config for name in required_configs):
                raise ValueError(f"{lr_schedule_name} requires {required_configs}")
            
            # Define RL Schedule
            lr = schedule(
                init_value=lr_schedule_config.init_value,
                peak_value=optim_config.learning_rate,
                warmup_steps=lr_schedule_config.warmup_steps,
                transition_steps=lr_schedule_config.transition_steps,
                decay_rate=lr_schedule_config.decay_rate,
                transition_begin=lr_schedule_config.transition_begin,)

        elif lr_schedule_name == "linear_schedule":
            # Check required configs are present
            required_configs = ["end_value", "transition_steps"]
            if not all(name in lr_schedule_config for name in required_configs):
                raise ValueError(f"{lr_schedule_name} requires {required_configs}")

            # Define LR Schedule
            lr = schedule(
                init_value=optim_config.learning_rate,
                end_value=lr_schedule_config.end_value,
                transition_steps=lr_schedule_config.transition_steps,
            )
        else:
            raise ValueError("Scheduler not supported")
    
    else:
        lr = optim_config.learning_rate

    optimizer = getattr(optax, optim_name)
    # optimizer = optax.inject_hyperparams(optimizer)

    use_nesterov = optim_config.get("nesterov", False)
    weight_decay = optim_config.get("weight_decay", None)

    absolute_clipping = optim_config.get("absolute_clipping", None)

    if optim_name == "sgd":
        optimizer = optax.inject_hyperparams(optax.sgd)(
            learning_rate=lr, momentum=optim_config.momentum, nesterov=use_nesterov
        )
        if weight_decay is not None:
            optimizer = optax.chain(
                optimizer, optax.additive_weight_decay(weight_decay, model_mask)
            )
        

    if optim_name == "adamw":
        # If adamw, weight_decay is a passable parameter.
        if weight_decay is None:
            raise ValueError("weight_decay must be specified for adamw")
        optimizer = optimizer(learning_rate=lr, weight_decay=weight_decay)

    if optim_name == "adam":
        optimizer = optimizer(learning_rate=lr)

    if absolute_clipping is not None and absolute_clipping > 0:
        
        optimizer = optax.chain(
            optax.clip_by_global_norm(absolute_clipping),
            optimizer
        )

    return optimizer


def get_lr(opt_state):
    if isinstance(opt_state, optax.InjectHyperparamsState):
        lr_to_log = opt_state.hyperparams["learning_rate"]
    elif isinstance(opt_state, tuple):
        for o_state in opt_state:
            # print('test', type(o_state))
            if isinstance(o_state, optax.InjectHyperparamsState):
                lr_to_log = o_state.hyperparams["learning_rate"]
            if isinstance(o_state, tuple):
                for o_state2 in o_state:
                    if isinstance(o_state2, optax.InjectHyperparamsState):
                        lr_to_log = o_state2.hyperparams["learning_rate"]
    
    return lr_to_log