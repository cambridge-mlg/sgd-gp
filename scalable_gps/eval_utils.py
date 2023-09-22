from typing import Callable, List, Optional

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from chex import Array

from scalable_gps.data import Dataset
from scalable_gps.linalg_utils import KvP
from scalable_gps.linear_model import loss_fn, error, regularizer
from scalable_gps.inducing_linear_model import i_loss_fn
from scalable_gps.utils import (
    ExactPredictionsTuple,
    ExactSamplesTuple,
    TargetTuple,
    revert_z_score,
)
from scalable_gps.optim_utils import get_uniform_idx_fn
from scalable_gps.utils import ExactPredictionsTuple, ExactSamplesTuple, TargetTuple, revert_z_score


def grad_var_fn(
    params: Array,
    grad_fn: Callable,
    idx: Array,
    features: Array,
    target_tuple: TargetTuple,
    num_evals: int = 100,
    key: chex.PRNGKey = jr.PRNGKey(12345),
):
    batch_size = idx.shape[0]
    n_train = params.shape[0]
    idx_fn = get_uniform_idx_fn(batch_size, n_train)

    @jax.jit
    @jax.vmap
    def batch_grad_fn(key):
        idx = idx_fn(0, key)
        grad_val = grad_fn(params, idx, features, target_tuple)

        return grad_val

    grad_var_key = jr.split(key, num_evals)
    grad_samples = batch_grad_fn(grad_var_key)
    grad_var = jnp.var(grad_samples, axis=0).mean()

    return grad_var


def hilbert_space_RMSE(x: Array, x_hat: Array, K: Array):
    return jnp.sqrt(jnp.mean((x - x_hat).T * (K @ (x - x_hat))))

def pred_space_RMSE(x: Array, x_hat: Array, K: Array):
    return jnp.sqrt(jnp.mean((K @ (x - x_hat)).T * (K @ (x - x_hat))))


def RMSE(
    x: Array, x_hat: Array, mu: Optional[Array] = None, sigma: Optional[Array] = None
):
    if mu is not None and sigma is not None:
        x = revert_z_score(x, mu, sigma)
        x_hat = revert_z_score(x_hat, mu, sigma)
    return jnp.sqrt(jnp.mean((x - x_hat) ** 2))


def mean_LLH(
    x: Array, loc: Array, variance: Array, mu: Optional[Array] = None, sigma: Optional[Array] = None
):
    """Calculate the log-likelihood of x given loc and scale."""
    scale = jnp.sqrt(variance)
    if mu is not None and sigma is not None:
        x = revert_z_score(x, mu, sigma)
        loc = revert_z_score(loc, mu, sigma)
        scale = revert_z_score(scale, 0.0, sigma)
    return jnp.mean(jax.scipy.stats.norm.logpdf(x, loc=loc, scale=scale))


def R2_score(
    x_hat: Array, x: Array, mu: Optional[Array] = None, sigma: Optional[Array] = None
):
    # TODO: implement check for zero denominator.
    if mu is not None and sigma is not None:
        x = revert_z_score(x, mu, sigma)
        x_hat = revert_z_score(x_hat, mu, sigma)
    
    y_diff_avg = jnp.average(x - x_hat, axis=0)
    numerator = jnp.average(
        (x - x_hat - y_diff_avg) ** 2, axis=0
    )

    y_true_avg = jnp.average(x_hat, axis=0)
    denominator = jnp.average((x_hat - y_true_avg) ** 2, axis=0)

    return jnp.mean(1. - numerator / denominator)



def get_eval_fn(
    metrics_list: List[str],
    train_ds: Dataset,
    test_ds: Dataset,
    kernel_fn: Callable,
    noise_scale: float,
    grad_fn: Optional[Callable] = None,
    metrics_prefix: str = "",
    exact_metrics: Optional[ExactPredictionsTuple] = None,
    exact_samples: Optional[ExactSamplesTuple] = None,
    vmap_and_pmap: bool = False,
):
    def _fn(params, idx, features, target_tuple):
        if not metrics_list:
            return {}
        
        # Calculate all quantities of interest here, and each metric_fn gets passed all quantities.
        if exact_metrics is not None:
            alpha_exact = exact_metrics.alpha
            y_pred_loc_exact = exact_metrics.y_pred_loc


        if exact_samples is not None and vmap_and_pmap:
            # Exact samples needs to be vmapped, so we can access vmap_idx
            vmap_idx = jax.lax.axis_index("sample")
            pmap_idx = jax.lax.axis_index("device")
            
            # Reshape exact_samples from (n_samples, ...) to (n_devices, n_samples_per_device, ...)
            n_devices = jax.device_count()
            n_samples = exact_samples.alpha_sample.shape[0]
            n_samples_per_device = n_samples // n_devices

            exact_samples_reshaped = jax.tree_map(
                lambda x: jnp.reshape(x, (n_devices, n_samples_per_device, -1)), exact_samples)
            
            alpha_exact = exact_samples_reshaped.alpha_sample[pmap_idx, vmap_idx]
            # y_pred_exact = (K(·)x(Kxx + Σ)^{−1} y) + f0(·) − K(·) (Kxx + Σ)^{−1} (f0(x) + ε0).
            y_pred_loc_exact = exact_samples_reshaped.posterior_sample[pmap_idx, vmap_idx]
            alpha_map = exact_samples_reshaped.alpha_map[pmap_idx, vmap_idx]
            f0_sample_test = exact_samples_reshaped.f0_sample_test[pmap_idx, vmap_idx]
            
            y_pred_loc_sgd = f0_sample_test + KvP(test_ds.x, train_ds.x, alpha_map - params, kernel_fn=kernel_fn)
            
            print(f'alpha_exact.shape: {alpha_exact.shape}')
        else:
            y_pred_loc_sgd = KvP(test_ds.x, train_ds.x, params, kernel_fn=kernel_fn)

        # TODO: Add normalised_test_rmse_Diff and test_rmse_Diff

        # Define all metric function calls here for now, refactor later.
        
        all_metrics = ['loss', 'err', 'reg', 'grad_var', 'test_rmse', 'normalised_test_rmse', 
                       'alpha_diff', 'alpha_rkhs_diff', 'y_pred_diff', "R2", "y_pred_test_diff"]
        def _get_metric(metric):
            if metric == "loss":
                return loss_fn(
                    params,
                    idx,
                    train_ds.x,
                    features,
                    target_tuple,
                    kernel_fn,
                    noise_scale,
                )
            elif metric == "err":
                return error(
                    params,
                    idx,
                    train_ds.x,
                    target_tuple.error_target,
                    kernel_fn
                )
            elif metric == "reg":
                return regularizer(
                    params,
                    features,
                    target_tuple.regularizer_target,
                    noise_scale
                )
            elif metric == "grad_var":
                return grad_var_fn(params, grad_fn, idx, features, target_tuple)
            elif metric == "test_rmse":
                return RMSE(
                    test_ds.y, y_pred_loc_sgd, mu=train_ds.mu_y, sigma=train_ds.sigma_y
                )
            elif metric == "normalised_test_rmse":
                return RMSE(test_ds.y, y_pred_loc_sgd)
            elif metric == "R2":
                return R2_score(test_ds.y, y_pred_loc_sgd)
            elif metric == "alpha_diff":
                return RMSE(alpha_exact, params)
            elif metric == "alpha_rkhs_diff":
                return hilbert_space_RMSE(
                    alpha_exact, params, K=kernel_fn(train_ds.x, train_ds.x)
                )
            elif metric == "y_pred_diff":
                # # TODO: right now we measure the difference between zero_mean posterior_samples, as alpha_map used for
                # # both y_pred_test and y_pred_exact is alpha_map of ExactGP, and gets cancelled out.
                # return RMSE(
                #     y_pred_loc_sgd,
                #     y_pred_loc_exact,
                #     mu=train_ds.mu_y,
                #     sigma=train_ds.sigma_y,
                # )
                return pred_space_RMSE(
                    alpha_exact, params, K=kernel_fn(train_ds.x, train_ds.x)
                )
            elif metric == "y_pred_test_diff":
                return RMSE(
                    y_pred_loc_sgd,
                    y_pred_loc_exact,
                    mu=train_ds.mu_y,
                    sigma=train_ds.sigma_y,
                )

        metrics_update_dict = {}

        # TODO: dont return N_steps dicts
        for metric in metrics_list:
            if metric in all_metrics:
                metrics_update_dict[f"{metrics_prefix}/{metric}"] = _get_metric(metric)

        return metrics_update_dict
    
    if vmap_and_pmap:
        return jax.jit(jax.pmap(
                jax.vmap(_fn, in_axes=(0, None, None, 0), axis_name='sample'),
                in_axes=(0, None, None, 0), axis_name='device'))
    return jax.jit(_fn)


def get_inducing_eval_fn(
    metrics_list: List[str],
    train_ds: Dataset,
    test_ds: Dataset,
    kernel_fn: Callable,
    noise_scale: float,
    grad_fn: Optional[Callable] = None,
    metrics_prefix: str = "",
    exact_metrics: Optional[ExactPredictionsTuple] = None,
    exact_samples: Optional[ExactSamplesTuple] = None,
    vmap_and_pmap: bool = False,
):
    del exact_samples, grad_fn
    def _fn(params, idx, features_x, features_z, target_tuple):
        # Calculate all quantities of interest here, and each metric_fn gets passed all quantities.

        if exact_metrics is not None:
            alpha_exact = exact_metrics.alpha
            y_pred_loc_exact = exact_metrics.y_pred_loc

        y_pred_loc_sgd = KvP(test_ds.x, train_ds.z, params, kernel_fn=kernel_fn)

        # TODO: Add normalised_test_rmse_Diff and test_rmse_Diff
        # Define all metric function calls here for now, refactor later.
        def _get_metric(metric):
            if metric == "loss":
                return i_loss_fn(
                    params,
                    idx,
                    train_ds.x,
                    train_ds.z,
                    features_x,
                    features_z,
                    target_tuple,
                    kernel_fn,
                    noise_scale,
                )[0]
            elif metric == "err":
                return i_loss_fn(
                    params,
                    idx,
                    train_ds.x,
                    train_ds.z,
                    features_x,
                    features_z,
                    target_tuple,
                    kernel_fn,
                    noise_scale,
                )[1]
            elif metric == "reg":
                return i_loss_fn(
                    params,
                    idx,
                    train_ds.x,
                    train_ds.z,
                    features_x,
                    features_z,
                    target_tuple,
                    kernel_fn,
                    noise_scale,
                )[2]
            elif metric == "test_rmse":
                return RMSE(
                    test_ds.y, y_pred_loc_sgd, mu=train_ds.mu_y, sigma=train_ds.sigma_y
                )
            elif metric == "normalised_test_rmse":
                return RMSE(test_ds.y, y_pred_loc_sgd)
            elif metric == "R2":
                return R2_score(test_ds.y, y_pred_loc_sgd)
            elif metric == "alpha_diff":
                return RMSE(alpha_exact, params)
            elif metric == "alpha_rkhs_diff":
                return hilbert_space_RMSE(
                    alpha_exact, params, K=kernel_fn(train_ds.z, train_ds.z)
                )
            elif metric == "y_pred_diff":
                # TODO: right now we measure the difference between zero_mean posterior_samples, as alpha_map used for
                # both y_pred_test and y_pred_exact is alpha_map of ExactGP, and gets cancelled out.
                return RMSE(
                    y_pred_loc_sgd,
                    y_pred_loc_exact,
                    mu=train_ds.mu_y,
                    sigma=train_ds.sigma_y,
                )

        metrics_update_dict = {}

        # TODO: dont return N_steps dicts
        for metric in metrics_list:
            metrics_update_dict[f"{metrics_prefix}/{metric}"] = _get_metric(metric)

        return metrics_update_dict

    if vmap_and_pmap:
        return jax.jit(jax.pmap(
                jax.vmap(_fn, in_axes=(0, None, None, 0), axis_name='sample'),
                in_axes=(0, None, None, 0), axis_name='device'))
    return jax.jit(_fn)


def get_exact_sample_tuples_fn(alpha_map):
    def _fn(alpha_sample, posterior_sample, f0_sample_test):
        return ExactSamplesTuple(
            alpha_sample=alpha_sample,
            posterior_sample=posterior_sample,
            f0_sample_test=f0_sample_test,
            alpha_map=alpha_map,
        )

    return jax.vmap(_fn)
