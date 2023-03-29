from functools import partial
from typing import Callable, List, Optional

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import ml_collections
import optax
from chex import Array
from data import Dataset
from linalg_utils import KvP
from linear_model import (
    error_grad_sample,
    regularizer_grad_sample,
)
from metrics import RMSE, grad_var_fn, hilbert_space_RMSE
from tqdm import tqdm
from utils import TargetTuple, ExactMetricsTuple, ExactSamplesTuple


# TODO: if for error_fn pmap and reg_fn pmap
def get_stochastic_gradient_fn(
    x: Array,
    target_tuple: TargetTuple,
    kernel_fn: Callable,
    noise_scale: float
):
    @jax.jit
    def _fn(params, idx, features):
        error_grad = error_grad_sample(
            params, idx, target_tuple.error_target, kernel_fn
        )
        regularizer_grad = regularizer_grad_sample(
            params,
            features,
            x,
            target_tuple.regularizer_target,
            noise_scale,
        )
        return error_grad + regularizer_grad

    return _fn


def get_update_fn(grad_fn: Callable, n_train: int, polyak_step_size: float):
    @partial(jax.jit, static_argnums=(3))
    def _fn(params, params_polyak, idx, features, optimizer, opt_state):
        grad = grad_fn(params, idx, features) / n_train

        updates, opt_state = optimizer.update(grad, opt_state)
        new_params = optax.apply_updates(params, updates)
        new_params_polyak = optax.incremental_update(
            new_params, params_polyak, step_size=polyak_step_size
        )

        return new_params, new_params_polyak, opt_state

    return _fn


def get_eval_fn(
    metrics: List[str],
    train_ds: Dataset,
    test_ds: Dataset,
    loss_fn: Callable,
    grad_fn: Callable,
    target_tuple: TargetTuple,
    kernel_fn: Callable,
    feature_fn: Callable,
    noise_scale: float,
    metrics_prefix: str = "",
    exact_metrics: Optional[ExactMetricsTuple] = None,
    exact_samples: Optional[ExactSamplesTuple] = None
):
    @jax.jit
    def _fn(params, idx, features):
        B, N = idx.shape[0], train_ds.N
        # Calculate all quantities of interest here, and each metric_fn gets passed all quantities.
        
        if exact_metrics is not None:
            alpha_exact, y_pred_exact, test_rmse_exact = exact_metrics
            y_pred_test = KvP(test_ds.x, train_ds.x, params, kernel_fn=kernel_fn)

        if exact_samples is not None:
            alpha_exact, y_pred_exact, test_rmse_exact, alpha_map = exact_samples
            y_pred_test = KvP(test_ds.x, train_ds.x, alpha_map - params, kernel_fn=kernel_fn)

        # Define all metric function calls here for now, refactor later.
        def _get_metric(metric):
            if metric == "loss":
                return loss_fn(params, idx, train_ds.x, features, target_tuple, kernel_fn, noise_scale)
            elif metric == "grad_var":
                return grad_var_fn(params, grad_fn, B, train_ds, feature_fn)
            elif metric == "test_rmse":
                return RMSE(test_ds.y, y_pred_test, mu=train_ds.mu_y, sigma=train_ds.sigma_y)
            elif metric == "alpha_diff":
                return RMSE(alpha_exact, params)
            elif metric == "alpha_rkhs_diff":
                return hilbert_space_RMSE(alpha_exact, params, K=kernel_fn(train_ds.x, train_ds.x))
            elif metric == "test_rmse_diff":
                return RMSE(_get_metric("test_rmse"), test_rmse_exact)
            elif metric == "y_pred_diff":
                return RMSE(y_pred_test, y_pred_exact)

        metrics_update_dict = {}

        # TODO: dont return N_steps dicts
        for metric in metrics:
            metrics_update_dict[f"{metrics_prefix}/{metric}"] = _get_metric(metric)

        return metrics_update_dict

    return _fn


def train(
    key: chex.PRNGKey, 
    config: ml_collections.ConfigDict, 
    update_fn: Callable, 
    eval_fn: Callable,
    feature_fn: Callable, 
    train_ds: Dataset,
    params: Array, 
    params_polyak: Array):

    aux = []
    optimizer = optax.sgd(
        learning_rate=config.learning_rate, momentum=config.momentum, nesterov=True
    )
    opt_state = optimizer.init(params)

    B, N = config.batch_size, params.shape[0]
    iterator = tqdm(range(config.iterations))
    for i in iterator:
        # perform update
        key, idx_key, feature_key = jr.split(key)
        idx = jr.randint(idx_key, shape=(B,), minval=0, maxval=N)
        features = feature_fn(feature_key, train_ds.x)

        params, params_polyak, opt_state = update_fn(
            params, params_polyak, idx, features, optimizer, opt_state
        )

        if i % config.eval_every == 0 and eval_fn is not None:
            eval_metrics = eval_fn(params_polyak, idx, features)
            # wandb.log(eval_metrics)
            aux.append(eval_metrics)

    return params_polyak, aux
