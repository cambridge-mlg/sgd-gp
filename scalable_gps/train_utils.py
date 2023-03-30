from typing import Callable, List, Optional

import chex
import jax
import ml_collections
import optax
from chex import Array
from data import Dataset
from linalg_utils import KvP
from metrics import RMSE, grad_var_fn, hilbert_space_RMSE
from utils import TargetTuple, ExactMetricsTuple, ExactSamplesTuple

# TODO: if for error_fn pmap and reg_fn pmap


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
        B = idx.shape[0]
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


# def train(
#     key: chex.PRNGKey, 
#     config: ml_collections.ConfigDict, 
#     update_fn: Callable, 
#     eval_fn: Callable,
#     feature_fn: Callable, 
#     train_ds: Dataset,
#     params: Array, 
#     params_polyak: Array):

#     aux = []
#     optimizer = optax.sgd(
#         learning_rate=config.learning_rate, momentum=config.momentum, nesterov=True
#     )
#     opt_state = optimizer.init(params)

#     B, N = config.batch_size, params.shape[0]
#     iterator = tqdm(range(config.iterations))
    
#     @jax.jit
#     def _get_idx_and_features(idx_key, feature_key):
#         idx = jr.randint(idx_key, shape=(B,), minval=0, maxval=N)
#         features = feature_fn(key=feature_key, x=train_ds.x, recompute=True)

#         return idx, features

#     for i in iterator:
#         # perform update
#         key, idx_key, feature_key = jr.split(key, 3)
        
#         # Calculate mini-batch specific idx and features here.
#         idx, features = _get_idx_and_features(idx_key, feature_key)

#         params, params_polyak, opt_state = update_fn(
#             params, params_polyak, idx, features, optimizer, opt_state
#         )

#         # if i % config.eval_every == 0 and eval_fn is not None:
#         #     eval_metrics = eval_fn(params_polyak, idx, features)
#         #     wandb.log(eval_metrics)
#         #     aux.append(eval_metrics)

#     return params_polyak, aux

import jax.random as jr


def get_train_loop_body(config, update_fn, feature_fn, train_ds, optimizer):
    
    def _train_loop_body(i, loop_vars):
        key, params, params_polyak, opt_state = loop_vars
        # key = progress_bar((i + 1, total_steps, total_steps // 10), key)
        idx_key, feature_key = jr.split(jr.fold_in(key, i), 2)
        # idx_key, feature_key = jr.split(key, 2)
        idx = jr.randint(idx_key, shape=(config.batch_size,), minval=0, maxval=train_ds.N)
        features = feature_fn(key=feature_key, x=train_ds.x, recompute=True)
        
        params, params_polyak, opt_state = update_fn(
            params, params_polyak, idx, features, optimizer, opt_state
        )

        # eval_fn(params_polyak, idx, features)
        # wandb.log(eval_metrics)

        return (key, params, params_polyak, opt_state)

    return jax.jit(_train_loop_body)


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
    
    n_eval_iterations = config.iterations // config.eval_every
    n_iterations_per_eval = config.iterations // n_eval_iterations
    
    print(f'Running {n_eval_iterations} eval iterations with {n_iterations_per_eval} iterations per eval.')
    
    body_fun = get_train_loop_body(config, update_fn, feature_fn, train_ds, optimizer)
    loop_vars = (key, params, params_polyak, opt_state)
    
    loop_vars = jax.lax.fori_loop(0, config.iterations, body_fun, loop_vars)

    _, _, params_polyak, _ = loop_vars
    return params_polyak, aux


from jax.experimental import host_callback


def _print_consumer(arg, transform):
    iter_num, num_samples = arg
    print(f"Iteration {iter_num:,} / {num_samples:,}")

@jax.jit
def progress_bar(arg, result):
    """
    Print progress of a scan/loop only if the iteration number is a multiple of the print_rate

    Usage: `carry = progress_bar((iter_num + 1, num_samples, print_rate), carry)`
    Pass in `iter_num + 1` so that counting starts at 1 and ends at `num_samples`

    """
    iter_num, num_samples, print_rate = arg
    result = jax.lax.cond(
        iter_num % print_rate==0,
        lambda _: host_callback.id_tap(_print_consumer, (iter_num, num_samples), result=result),
        lambda _: result,
        operand=None)
    return result