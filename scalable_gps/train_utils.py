import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from tqdm import tqdm

import wandb
from .linalg_utils import KvP
from .linear_model import (
    error_grad_sample,
    regularizer_grad_sample,
)
from .metrics import RMSE, grad_var_fn
from functools import partial


# TODO: if for error_fn pmap and reg_fn pmap
def get_stochastic_gradient_fn(
    x,
    target_tuple,
    kernel_fn,
    feature_fn,
    batch_size,
    num_features,
    noise_scale,
    recompute_features=True,
):
    @jax.jit
    def _fn(params, key):
        error_key, regularizer_key = jr.split(key)
        error_grad = error_grad_sample(
            params, error_key, batch_size, x, target_tuple.error_target, kernel_fn
        )
        regularizer_grad = regularizer_grad_sample(
            params,
            regularizer_key,
            num_features,
            x,
            target_tuple.regularizer_target,
            feature_fn,
            noise_scale,
            recompute_features=recompute_features,
        )
        return error_grad + regularizer_grad

    return _fn


def get_update_fn(grad_fn, n_train, polyak_step_size):
    @partial(jax.jit, static_argnums=(3))
    def _fn(key, params, params_polyak, optimizer, opt_state):
        grad = grad_fn(params, key) / n_train

        updates, opt_state = optimizer.update(grad, opt_state)
        new_params = optax.apply_updates(params, updates)
        new_params_polyak = optax.incremental_update(
            new_params, params_polyak, step_size=polyak_step_size
        )

        return new_params, new_params_polyak, opt_state

    return _fn


def get_eval_fn(
    metrics,
    train_ds,
    test_ds,
    loss_fn,
    grad_fn,
    target_tuple,
    kernel_fn,
    noise_scale,
    metrics_prefix="",
    # TODO: this should be a named tuple
    compare_exact_vals=None,
):
    def _fn(params):

        # Calculate all quantities of interest here, and each metric_fn gets passed all quantities.
        y_pred_test = KvP(test_ds.x, train_ds.x, params, kernel_fn=kernel_fn)

        if compare_exact_vals is not None:
            alpha_exact, y_pred_exact, test_rmse_exact, _, _ = compare_exact_vals

        # Define all metric function calls here for now, refactor later.
        def _get_metric(metric):
            if metric == "loss":
                K_train = kernel_fn(train_ds.x, train_ds.x)
                return loss_fn(params, target_tuple, K_train, noise_scale=noise_scale)
            elif metric == "grad_var":
                return grad_var_fn(params, grad_fn)
            elif metric == "test_rmse":
                return RMSE(
                    test_ds.y, y_pred_test, mu=train_ds.mu_y, sigma=train_ds.sigma_y
                )
            elif metric == "alpha_diff":
                return RMSE(alpha_exact, params)
            # TODO: add kernel weighed alpha diff

            elif metric == "test_rmse_diff":
                return RMSE(_get_metric("test_rmse"), test_rmse_exact)
            elif metric == "y_pred_diff":
                return RMSE(y_pred_test, y_pred_exact)

        metrics_update_dict = {}

        # TODO: dont return N_steps dicts
        for metric in metrics:
            metrics_update_dict[f"{metrics_prefix}/{metric}"] = _get_metric(metric)

        # TODO: this should not be called from here since it will break this
        # function if not called from inside main.py.
        # wandb.log(metrics_update_dict)

        return metrics_update_dict

    return _fn


def get_sampling_eval_fn(
    metrics,
    train_ds,
    test_ds,
    prior_fn_sample_test,
    params_map,
    loss_fn,
    grad_fn,
    target_tuple,
    kernel_fn,
    noise_scale,
    metrics_prefix="",
    compare_exact_vals=None,
):
    def _fn(params):

        # Calculate all quantities of interest here.
        K_train = kernel_fn(train_ds.x, train_ds.x)
        y_pred_sample = prior_fn_sample_test + KvP(
            test_ds.x, train_ds.x, params_map - params, kernel_fn=kernel_fn
        )

        if compare_exact_vals is not None:
            _, _, _, alpha_sample_exact, y_pred_sample_exact = compare_exact_vals

        # Define all metric function calls here for now, refactor later.
        def _get_metric(metric):
            if metric == "loss":
                return loss_fn(params, target_tuple, K_train, noise_scale=noise_scale)
            elif metric == "grad_var":
                return grad_var_fn(params, grad_fn)
            elif metric == "test_rmse":
                return RMSE(
                    test_ds.y, y_pred_sample, mu=train_ds.mu_y, sigma=train_ds.sigma_y
                )
            elif metric == "alpha_sample_diff":
                return RMSE(alpha_sample_exact, params)
            elif metric == "y_pred_diff":
                return RMSE(y_pred_sample_exact, y_pred_sample)
            elif metric == "loss_diff":
                exact_loss = loss_fn(
                    alpha_sample_exact, target_tuple, K_train, noise_scale=noise_scale
                )
                return _get_metric("loss") - exact_loss
            elif metric == "test_rmse_diff":
                test_rmse_exact = RMSE(
                    test_ds.y,
                    y_pred_sample_exact,
                    mu=train_ds.mu_y,
                    sigma=train_ds.sigma_y,
                )
                return RMSE(_get_metric("test_rmse"), test_rmse_exact)

        metrics_update_dict = {}

        for metric in metrics:
            metrics_update_dict[f"{metrics_prefix}/{metric}"] = _get_metric(metric)

        wandb.log(metrics_update_dict)

        return metrics_update_dict

    return _fn


def train(key, config, update_fn, eval_fn, params, params_polyak):

    aux = []
    optimizer = optax.sgd(
        learning_rate=config.learning_rate, momentum=config.momentum, nesterov=True
    )
    opt_state = optimizer.init(params)

    iterator = tqdm(range(config.iterations))
    for i in iterator:
        # perform update
        key, subkey = jr.split(key)
        params, params_polyak, opt_state = update_fn(
            subkey, params, params_polyak, optimizer, opt_state
        )

        if i % config.eval_every == 0 and eval_fn is not None:
            aux.append(eval_fn(params_polyak))

    return params_polyak, aux
