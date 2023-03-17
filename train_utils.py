
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from tqdm import tqdm

import wandb
from linalg_utils import calc_Kstar_v
from linear_model import (
    error_grad_sample,
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


def train(key, config, update_fn, eval_fn, params, params_polyak):

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
