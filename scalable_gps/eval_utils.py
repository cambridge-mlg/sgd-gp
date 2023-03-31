from typing import Callable, List, Optional

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from chex import Array
from data import Dataset
from linalg_utils import KvP
from linear_model import loss_fn
from utils import ExactMetricsTuple, ExactSamplesTuple, revert_z_score


def grad_var_fn(
    params: Array,
    grad_fn: Callable,
    batch_size: int,
    train_ds: Dataset,
    feature_fn: Callable, 
    num_evals: int = 100,
    key: chex.PRNGKey = jr.PRNGKey(12345),
):
    # TODO (jandylin): Rewrite this to work with our new grad_fn and idx formulation?
    B, N = batch_size, train_ds.N
    @jax.jit
    def _compute_grad(single_key):
        idx_key, feature_key = jr.split(single_key)
        idx = jr.randint(idx_key, shape=(B,), minval=0, maxval=N)
        features = feature_fn(key=feature_key)
        grad_val = grad_fn(params, idx, features)

        return grad_val
    grad_var_key = jr.split(key, num_evals)
    grad_samples = jax.vmap(_compute_grad)(grad_var_key)
    grad_var = jnp.var(grad_samples, axis=0).mean()

    return grad_var


def hilbert_space_RMSE(x: Array, x_hat: Array, K: Array):
    return jnp.sqrt(jnp.mean((x - x_hat) * (K @ (x - x_hat))))


def RMSE(
    x: Array, x_hat: Array, mu: Optional[Array] = None, sigma: Optional[Array] = None
):
    if mu is not None and sigma is not None:
        x = revert_z_score(x, mu, sigma)
        x_hat = revert_z_score(x_hat, mu, sigma)
    return jnp.sqrt(jnp.mean((x - x_hat) ** 2))


def get_eval_fn(
    metrics_list: List[str],
    train_ds: Dataset,
    test_ds: Dataset,
    grad_fn: Callable,
    kernel_fn: Callable,
    feature_fn: Callable,
    noise_scale: float,
    metrics_prefix: str = "",
    exact_metrics: Optional[ExactMetricsTuple] = None,
    exact_samples: Optional[ExactSamplesTuple] = None,
    vmap: bool = False,
):
    def _fn(params, idx, features, target_tuple):
        idx.shape[0]
        # Calculate all quantities of interest here, and each metric_fn gets passed all quantities.
        
        if exact_metrics is not None:
            alpha_exact, y_pred_exact, test_rmse_exact = exact_metrics
            y_pred_test = KvP(test_ds.x, train_ds.x, params, kernel_fn=kernel_fn)

        if exact_samples is not None and vmap:
            # Exact samples needs to be vmapped, so we can access vmap_idx
            vmap_idx = jax.lax.axis_index("sample")
            

            alpha_exact = exact_samples.alpha_sample[vmap_idx]
            # y_pred_exact = (K(·)x(Kxx + Σ)^{−1} y) + f0(·) − K(·) (Kxx + Σ)^{−1} (f0(x) + ε0).
            y_pred_exact = exact_samples.posterior_sample[vmap_idx]
            alpha_map = exact_samples.alpha_map[vmap_idx]
            f0_sample_test = exact_samples.f0_sample_test[vmap_idx]
            
            y_pred_test = f0_sample_test + KvP(test_ds.x, train_ds.x, alpha_map - params, kernel_fn=kernel_fn)
            test_rmse_exact = RMSE(test_ds.y, y_pred_exact, mu=train_ds.mu_y, sigma=train_ds.sigma_y)
            
            print(f'alpha_exact.shape: {alpha_exact.shape}')

        # Define all metric function calls here for now, refactor later.
        def _get_metric(metric):
            if metric == "loss":
                return loss_fn(params, idx, train_ds.x, features, target_tuple, kernel_fn, noise_scale)[0]
            elif metric == "err":
                return loss_fn(params, idx, train_ds.x, features, target_tuple, kernel_fn, noise_scale)[1]
            elif metric == "reg":
                return loss_fn(params, idx, train_ds.x, features, target_tuple, kernel_fn, noise_scale)[2]
            # TODO (jandylin): Can you rewrite this to match our new grad_fn etc. API?
            # elif metric == "grad_var":
            #     return grad_var_fn(params, grad_fn, B, train_ds, feature_fn)
            elif metric == "test_rmse":
                return RMSE(test_ds.y, y_pred_test, mu=train_ds.mu_y, sigma=train_ds.sigma_y)
            elif metric == "alpha_diff":
                return RMSE(alpha_exact, params)
            elif metric == "alpha_rkhs_diff":
                return hilbert_space_RMSE(alpha_exact, params, K=kernel_fn(train_ds.x, train_ds.x))
            elif metric == "test_rmse_diff":
                # TODO: for sampling this is technically wrong as we always use exact_gp.alpha_map as mean.
                return RMSE(_get_metric("test_rmse"), test_rmse_exact)
            elif metric == "y_pred_diff":
                # TODO: right now we measure the difference between zero_mean posterior_samples, as alpha_map used for
                # both y_pred_test and y_pred_exact is alpha_map of ExactGP, and gets cancelled out.
                return RMSE(y_pred_test, y_pred_exact, mu=train_ds.mu_y, sigma=train_ds.sigma_y)

        metrics_update_dict = {}

        # TODO: dont return N_steps dicts
        for metric in metrics_list:
            metrics_update_dict[f"{metrics_prefix}/{metric}"] = _get_metric(metric)

        return metrics_update_dict

    if vmap:
        return jax.jit(jax.vmap(_fn, in_axes=(0, None, None, 0), axis_name='sample'))
    return jax.jit(_fn)


def get_exact_sample_tuples_fn(alpha_map):
    
    def _fn(alpha_sample, posterior_sample, f0_sample_test):
        return ExactSamplesTuple(
            alpha_sample=alpha_sample, 
            posterior_sample=posterior_sample, 
            f0_sample_test=f0_sample_test,
            alpha_map=alpha_map)
    
    return jax.vmap(_fn)