from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from tqdm import tqdm

import wandb
from data import Dataset
from linalg_utils import calc_Kstar_v, solve_K_inv_v
from linear_model import (
    draw_prior_function_sample,
    draw_prior_noise_sample,
    error_grad_sample,
    loss_fn,
    regularizer_grad_sample,
)
from utils import RMSE


def get_stochastic_gradient_fn(x, target_tuple, kernel_fn, feature_fn, batch_size, num_features, noise_scale):
    
    error_target, regularizer_target = target_tuple
    @jax.jit
    def _fn(params, key):
        error_key, regularizer_key = jr.split(key)
        error_grad = error_grad_sample(params, error_key, batch_size, x, error_target, kernel_fn)
        regularizer_grad = regularizer_grad_sample(params, regularizer_key, num_features, x, regularizer_target, feature_fn)
        
        return error_grad + (noise_scale ** 2) * regularizer_grad
    
    return _fn


def get_update_fn(grad_fn, n_train, polyak_step_size):
    
    @jax.jit
    def _fn(key, params, params_polyak, optimizer, opt_state):
        grad = grad_fn(params, key) / n_train
        
        updates, opt_state = optimizer.update(grad, opt_state)
        new_params = optax.apply_updates(params, updates)
        new_params_polyak = optax.incremental_update(new_params, params_polyak, step_size=polyak_step_size)
        
        return new_params, new_params_polyak, opt_state
    
    return _fn


def get_eval_fn(train_ds, test_ds, loss_fn, grad_fn, targets, kernel_fn, noise_scale, alpha_exact=None):
    
    def _fn(params, K):
        loss = loss_fn(params, targets, K, noise_scale=noise_scale)
        # compute trace statistics

        calc_Kstar_v(test_ds.x, train_ds.x, params, kernel_fn)
        alpha_rmse = RMSE(alpha_exact, params)
        test_rmse = RMSE(test_ds.y, train_ds.y, mu=train_ds.mu_y, sigma=train_ds.sigma_y)

        grad_var_key = jr.split(jr.PRNGKey(12345), 100)
        grad_samples = jax.vmap(grad_fn, (None, 0))(params, grad_var_key)
        grad_var = jnp.var(grad_samples, axis=0).mean()
        
        metrics_update_dict = {
            'loss': loss,
            'grad_var': grad_var,
            'alpha_rmse': alpha_rmse,
            'test_rmse': test_rmse,}

        wandb.log({k: v[-1] for k, v in metrics_update_dict.items()})

        return metrics_update_dict
    
    return _fn


def compute_optimised_solution(key, config, train_ds, test_ds, kernel_fn, feature_fn, alpha_exact, **feature_kwargs):

    # Calculate the gradient function
    target_tuple = (train_ds.y, jnp.zeros_like(train_ds.y))
    grad_fn = get_stochastic_gradient_fn(
        train_ds.x, target_tuple, kernel_fn, feature_fn, config.batch_size, config.num_features, config.noise_scale)
    
    # Define the gradient update function
    update_fn = get_update_fn(grad_fn, train_ds.N, config.polyak)

    # @jax.jit
    eval_fn = get_eval_fn(train_ds, test_ds, loss_fn, grad_fn, target_tuple, kernel_fn, config.noise_scale, alpha_exact)

    # Initialise alpha and alpha_polyak
    alpha = jnp.zeros((train_ds.N,))
    alpha_polyak = jnp.zeros((train_ds.N,))
    
    alpha_polyak, aux = _train(
        key, config, update_fn, eval_fn, alpha, alpha_polyak)
    
    return alpha_polyak, aux


def compute_posterior_sample(key, config, train_ds, test_ds, kernel_fn, feature_fn, alpha_exact, **feature_kwargs):
    
    # Draw prior function sample evaluated at the train and test data
    feature_key, prior_fn_key, prior_noise_key, key = jr.split(key, 4)
    prior_function_sample_train = draw_prior_function_sample(
        feature_key, prior_fn_key, config.num_features, train_ds.x, feature_fn)
    # prior_function_sample_test = draw_prior_function_sample(
    #     feature_key, prior_fn_key, config.num_features, test_ds.x, feature_fn)
    
    # draw prior noise sample
    prior_noise_sample = draw_prior_noise_sample(prior_noise_key, train_ds.N, noise_scale=config.noise_scale)
    
    # Depending on the three types of losses we can compute the gradient of the loss function accordingly
    target_tuple = None
    if config.loss_type == 1:
        target_tuple = (prior_function_sample_train + prior_noise_sample, jnp.zeros_like(train_ds.y))
    elif config.loss_type == 2:
        target_tuple = (prior_function_sample_train, prior_noise_sample)
    elif config.loss_type == 3:
        target_tuple = (jnp.zeros_like(train_ds.y), prior_function_sample_train + prior_noise_sample)
    
    
    grad_fn = get_stochastic_gradient_fn(
        train_ds.x, target_tuple, kernel_fn, feature_fn, config.batch_size, config.num_features, config.noise_scale)
    
    # Define the gradient update function
    update_fn = get_update_fn(grad_fn, train_ds.N, config.polyak)
    
    eval_fn = None
    
    # Initialise alpha and alpha_polyak
    alpha = jnp.zeros((train_ds.N,))
    alpha_polyak = jnp.zeros((train_ds.N,))
    
    alpha_polyak, aux = _train(key, config, update_fn, eval_fn, alpha, alpha_polyak)
        
    



def _train(key, config, update_fn, eval_fn, params, params_polyak):

    aux = []
    optimizer = optax.sgd(learning_rate=config.learning_rate, momentum=config.momentum, nesterov=True)
    opt_state = optimizer.init(params)

    iterator = tqdm(range(config.iterations))
    for i in iterator:
        # perform update
        key, subkey = jr.split(key)
        opt_state, params, params_polyak = update_fn(subkey, params, params_polyak, optimizer, opt_state)

        if i % config.eval_every == 0 and eval_fn is not None:
            aux.append(eval_fn(params_polyak))

    return params_polyak, aux
    


def compute_exact_solution(
    train_ds: Dataset, test_ds: Dataset, kernel_fn: Callable, noise_scale: float):

    # Compute Kernel exactly
    K = kernel_fn(train_ds.x, train_ds.x)
    
    # Compute the representer weights by solving alpha = (K + sigma^2 I)^{-1} y
    alpha_exact = solve_K_inv_v(K, train_ds.y, noise_scale=noise_scale)
    
    # Calculate the predictions on the test set by K(x_test, x_train) @ alpha
    y_pred_exact = calc_Kstar_v(alpha_exact, test_ds.x, train_ds.x, kernel_fn=kernel_fn)
    
    # Calculate the test RMSE
    test_rmse_exact = RMSE(y_pred_exact, test_ds.y, mu=train_ds.mu_y, sigma=train_ds.sigma_y)
    
    return alpha_exact, y_pred_exact, test_rmse_exact