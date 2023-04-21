import time
from typing import Callable, List, Optional

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import ml_collections
import optax
import wandb
from chex import Array
from tqdm import tqdm

from scalable_gps import eval_utils, optim_utils, sampling_utils
from scalable_gps.data import Dataset
from scalable_gps.eval_utils import LLH, RMSE
from scalable_gps.kernels import Kernel
from scalable_gps.linalg_utils import KvP, pivoted_cholesky, solve_K_inv_v
from scalable_gps.linear_model import marginal_likelihood
from scalable_gps.optim_utils import get_lr_and_schedule
from scalable_gps.utils import ExactPredictionsTuple, HparamsTuple, TargetTuple, get_gpu_or_cpu_device


# TODO: Split file into three files within a models folder.
class GPModel:
    def __init__(self, noise_scale: float, kernel: Kernel):
        self.alpha = None
        self.y_pred = None
        self.K = None
        self.kernel = kernel
        self.noise_scale = noise_scale

    def compute_representer_weights(self):
        raise NotImplementedError("compute_representer_weights() must be implemented in the derived class.")

    def predictive_mean(self, train_ds: Dataset, test_ds: Dataset, recompute: bool = True) -> Array:
        if self.alpha is None:
            raise ValueError("alpha is None. Please call compute_representer_weights() first.")
        if recompute or self.y_pred is None:
            self.y_pred = KvP(test_ds.x, train_ds.x, self.alpha, kernel_fn=self.kernel.kernel_fn)

        return self.y_pred  # (N_test, 1)

    # TODO: Biased: use the method that double counts the diagonal (first paragraph of page 28 of https://arxiv.org/pdf/2210.04994.pdf
    # TODO: Unbiased: use a mixture of isotropic Gaussian likelihood with each mixture component's mean being centred at a sample. Then we can compute joint likelihoods as in the "EFFICIENT κ-ADIC SAMPLING" section on page 26 of https://arxiv.org/pdf/2210.04994.pdf
    def predictive_variance_samples(
        self, zero_mean_posterior_samples: Array, return_marginal_variance: bool = True) -> Array:
        """Compute MC estimate of posterior variance of the test points using zero mean samples from posterior."""
        # zero_mean_posterior_samples = (N_samples, N_test)
        if return_marginal_variance:
            variance = jnp.mean(zero_mean_posterior_samples ** 2, axis=0)  # (N_test, 1)
        else:
            n_samples = zero_mean_posterior_samples.shape[0]
            variance = zero_mean_posterior_samples.T @ zero_mean_posterior_samples / n_samples
    
        return variance

    def get_prior_samples_fn(self, n_train, L, use_rff: bool=False):
        """Vmap factory function for sampling from the prior."""
        # fn(keys) -> prior_samples
        def _fn(key):
            prior_sample_key, sample_key = jr.split(key)
            f0_sample_train, f0_sample_test = sampling_utils.draw_f0_sample(
                    prior_sample_key, n_train, L, use_rff=use_rff)
            eps0_sample = sampling_utils.draw_eps0_sample(
                sample_key, n_train, noise_scale=self.noise_scale)

            return f0_sample_train, f0_sample_test, eps0_sample

        return jax.jit(jax.vmap(_fn))

    def get_posterior_samples_fn(self, train_ds, test_ds, zero_mean: bool = True):
        """Vmap factory function for computing the zero mean posterior from sample."""

        def _fn(alpha_sample, f0_sample_test):
            zero_mean_posterior_sample = sampling_utils.compute_posterior_fn_sample(
                train_ds, 
                test_ds, 
                alpha_sample, 
                self.alpha, 
                f0_sample_test, 
                self.kernel.kernel_fn, 
                zero_mean=zero_mean)

            return zero_mean_posterior_sample

        return jax.jit(jax.vmap(_fn))

    def get_feature_fn(self, train_ds: Dataset, n_features: int, recompute: bool):
        """Factory function that wraps feature_fn so that it is jittable."""
        def _fn(key):
            return self.kernel.feature_fn(
                key, 
                n_features=n_features, 
                recompute=recompute, 
                x=train_ds.x)
        
        return jax.jit(_fn)    


class ExactGPModel(GPModel):

    def compute_representer_weights(self, train_ds: Dataset) -> Array:
        """Compute the representer weights alpha by solving alpha = (K + sigma^2 I)^{-1} y"""

        # Compute Kernel exactly
        self.K = self.kernel.kernel_fn(train_ds.x, train_ds.x) if self.K is None else self.K

        # Compute the representer weights by solving alpha = (K + sigma^2 I)^{-1} y
        self.alpha = solve_K_inv_v(self.K, train_ds.y, noise_scale=self.noise_scale)

        return self.alpha

    def predictive_variance(
        self, train_ds: Dataset, test_ds: Dataset, return_marginal_variance: bool = True) -> Array:
        """Compute the posterior variance of the test points."""
        K_test = self.kernel.kernel_fn(test_ds.x, test_ds.x)  # N_test, N_test
        K_train_test = self.kernel.kernel_fn(train_ds.x, test_ds.x)  # N_train, N_test
        # Compute Kernel exactly
        self.K = self.kernel.kernel_fn(train_ds.x, train_ds.x) if self.K is None else self.K

        K_inv_K_train_test = solve_K_inv_v(self.K, K_train_test, noise_scale=self.noise_scale)
    
        variance = K_test - K_train_test.T @ K_inv_K_train_test
        
        if return_marginal_variance:
            return jnp.diag(variance)  # (N_test, 1)
        return variance  # (N_test, N_test)

    def get_alpha_samples_fn(self):
        """Vmap factory function that returns a function that computes alpha samples from f0 and eps0 samples."""
        def _fn(f0_sample_train, eps0_sample):
            # (K + noise_scale**2 I)^{-1} (f0_sample_train + eps0_sample)
            alpha_sample = solve_K_inv_v(
                self.K,
                f0_sample_train + eps0_sample,
                noise_scale=self.noise_scale,
            )
            return alpha_sample

        return jax.jit(jax.vmap(_fn))
    
    def compute_posterior_samples(
        self, 
        key: chex.PRNGKey, 
        n_samples: int, 
        train_ds: Dataset, 
        test_ds: Dataset, 
        use_rff: bool = True,
        n_features: int = 0,
        chol_eps: float = 1e-5,
        L: Optional[Array] = None, 
        zero_mean: bool = True):
        """Computes n_samples posterior samples, and returns posterior_samples along with alpha_samples."""
        prior_covariance_key, prior_samples_key, samples_optim_key = jr.split(key, 3)
    
        if L is None:
            L = sampling_utils.compute_prior_covariance_factor(
                prior_covariance_key, 
                train_ds, 
                test_ds, 
                self.kernel.kernel_fn, 
                self.kernel.feature_fn,
                use_rff=use_rff, 
                n_features=n_features, 
                chol_eps=chol_eps)

        # Get vmapped functions for sampling from the prior and computing the posterior.
        compute_prior_samples_fn = self.get_prior_samples_fn(train_ds.N, L, use_rff)
        compute_alpha_samples_fn = self.get_alpha_samples_fn()
        compute_posterior_samples_fn = self.get_posterior_samples_fn(train_ds, test_ds, zero_mean)

        # Call the vmapped functions
        f0_samples_train, f0_samples_test, eps0_samples = compute_prior_samples_fn(
            jr.split(prior_samples_key, n_samples))  # (n_samples, n_train), (n_samples, n_test), (n_samples, n_train)

        alpha_samples = compute_alpha_samples_fn(f0_samples_train, eps0_samples)  # (n_samples, n_train)
        
        posterior_samples = compute_posterior_samples_fn(alpha_samples, f0_samples_test)  # (n_samples, n_test)

        chex.assert_shape(posterior_samples, (n_samples, test_ds.N))
        chex.assert_shape(alpha_samples, (n_samples, train_ds.N))

        return posterior_samples, alpha_samples

    def get_mll_loss_fn(self, train_ds: Dataset, kernel_fn: Callable, transform: Optional[Callable] = None):
        """Factory function that wraps mll_loss_fn so that it is jittable."""
        def _fn(log_hparams):

            return -marginal_likelihood(train_ds.x, train_ds.y, kernel_fn, hparams_tuple=log_hparams, transform=transform)
        
        return jax.jit(_fn, device=get_gpu_or_cpu_device())
    
    def get_mll_update_fn(self, mll_loss_fn, optimizer):
        """Factory function that wraps mll_update_fn so that it is jittable."""
        def _fn(log_hparams, opt_state):
            value, grad = jax.value_and_grad(mll_loss_fn)(log_hparams)
            # print(grad)
            updates, opt_state = optimizer.update(grad, opt_state)
            return value, optax.apply_updates(log_hparams, updates), opt_state
        
        return jax.jit(_fn)

    def compute_mll_optim(
        self, 
        init_hparams: HparamsTuple, 
        train_ds: Dataset, 
        config: ml_collections.ConfigDict, 
        test_ds,
        full_train_ds: Optional[Dataset] = None, 
        transform: Optional[Callable] = None, 
        perform_eval: bool = True):
        
        log_hparams = init_hparams

        optimizer = optax.adam(learning_rate=config.learning_rate)
        opt_state = optimizer.init(log_hparams)
        
        loss_fn = self.get_mll_loss_fn(train_ds, self.kernel.kernel_fn, transform=transform)
        update_fn = self.get_mll_update_fn(loss_fn, optimizer)
        
        iterator = tqdm(range(config.iterations))
        for i in iterator:
            loss_val, log_hparams, opt_state = update_fn(log_hparams, opt_state)

            hparams = log_hparams
            if transform is not None:
                hparams = HparamsTuple(
                    length_scale=transform(log_hparams.length_scale),
                    signal_scale=transform(log_hparams.signal_scale),
                    noise_scale=transform(log_hparams.noise_scale),)
                
            # TODO: Cleanup eval if needed.
            ############################### EVAL METRICS ##################################
            # Populate evaluation metrics etc.
            if perform_eval and ((i == 0) or ((i + 1) % config.eval_every == 0) or (i == (config.iterations - 1))):
                eval_train_ds = full_train_ds if full_train_ds is not None else train_ds
                K = self.kernel.kernel_fn(
                    eval_train_ds.x, eval_train_ds.x, length_scale=hparams.length_scale, signal_scale=hparams.signal_scale)

                # Compute the representer weights by solving alpha = (K + sigma^2 I)^{-1} y
                alpha = solve_K_inv_v(K, eval_train_ds.y, noise_scale=hparams.noise_scale)

                y_pred_test = KvP(
                    test_ds.x, eval_train_ds.x, alpha, kernel_fn=self.kernel.kernel_fn, 
                    length_scale=hparams.length_scale, signal_scale=hparams.signal_scale)
            
                test_rmse = RMSE(test_ds.y, y_pred_test, mu=eval_train_ds.mu_y, sigma=eval_train_ds.sigma_y)
                
                normalised_test_rmse = RMSE(test_ds.y, y_pred_test)
                
                iterator.set_description(f"Loss: {loss_val:.4f}")
                eval_metrics = {
                    "mll": -loss_val / eval_train_ds.N, 
                    "signal_scale": hparams.signal_scale, 
                    "length_scale": hparams.length_scale,
                    "noise_scale": hparams.noise_scale,
                    "test_rmse": test_rmse,
                    "normalised_test_rmse": normalised_test_rmse,}

                if wandb.run is not None:
                    wandb.log({**eval_metrics, **{'mll_train_step': i}})
            #########################################################################

        print("Final hyperparameters: ", hparams)
        
        return hparams

class SGDGPModel(GPModel):
    def __init__(self, noise_scale: float, kernel: Kernel, **kwargs):
        super().__init__(noise_scale=noise_scale, kernel=kernel, **kwargs)

    def compute_representer_weights(
        self,
        key: chex.PRNGKey,
        train_ds: Dataset,
        test_ds: Dataset,
        config: ml_collections.ConfigDict,
        metrics_list: List[str],
        metrics_prefix: str = "",
        exact_metrics: Optional[ExactPredictionsTuple] = None,
    ):
        """Compute the representer weights alpha by solving alpha = (K + sigma^2 I)^{-1} y using SGD."""
        target_tuple = TargetTuple(error_target=train_ds.y, regularizer_target=jnp.zeros_like(train_ds.y))
        
        # optimizer = optax.sgd(learning_rate=config.learning_rate, momentum=config.momentum, nesterov=True)
        
        optimizer = get_lr_and_schedule(
            "sgd", config, config.lr_schedule_name, config.lr_schedule_config)

        # Define the gradient function
        grad_fn = optim_utils.get_stochastic_gradient_fn(train_ds.x, self.kernel.kernel_fn, self.noise_scale)
        update_fn = optim_utils.get_update_fn(grad_fn, optimizer, config.polyak, vmap=False)
        feature_fn = self.get_feature_fn(train_ds, config.n_features_optim, config.recompute_features)
        idx_fn = optim_utils.get_idx_fn(config.batch_size, train_ds.N, config.iterative_idx, vmap=False)
        
        eval_utils.get_eval_fn(
            metrics_list,
            train_ds,
            test_ds,
            self.kernel.kernel_fn,
            self.noise_scale,
            feature_fn=feature_fn,
            grad_fn=grad_fn,
            metrics_prefix=metrics_prefix,
            exact_metrics=exact_metrics
        )

        def _get_cond_fn(time_budget_in_seconds, iterations, eval_every_in_seconds, eval_every):
            def _loop_cond_fn(time_elapsed, iter_counter):
                if time_budget_in_seconds > 0.:
                    return time_elapsed < time_budget_in_seconds
                else:
                    return iter_counter < iterations
            
            def _eval_cond_fn(time_elapsed, iter_counter, eval_counter):
                if time_budget_in_seconds > 0.:
                    return time_elapsed >= (eval_counter * eval_every_in_seconds)
                else:
                    return (iter_counter % eval_every == 0)

            if time_budget_in_seconds > 0.:
                pbar = tqdm(desc="Time: ", total=time_budget_in_seconds, unit="s")
            else:
                pbar = tqdm(desc="Iters: ", total=iterations)
            
            def _tqdm_update_fn(time_delta):
                pbar.update(time_delta if time_budget_in_seconds > 0. else 1)

            return jax.jit(_loop_cond_fn), jax.jit(_eval_cond_fn), pbar, _tqdm_update_fn
        
        loop_cond_fn, eval_cond_fn, pbar, tqdm_update_fn = _get_cond_fn(
            config.time_budget_in_seconds, config.iterations, config.eval_every_in_seconds, config.eval_every)
            
        # Initialise alpha and alpha_polyak
        alpha, alpha_polyak = jnp.zeros((train_ds.N,)), jnp.zeros((train_ds.N,))

        opt_state = optimizer.init(alpha)

        aux = []
        
        time_elapsed = 0.0
        iter_counter = 0
        eval_counter = 0
        
        with pbar:
            while loop_cond_fn(time_elapsed, iter_counter):
                time_before_update = time.time()
                key, idx_key, feature_key = jr.split(key, 3)
                features = feature_fn(feature_key)
                idx = idx_fn(iter_counter, idx_key)

                alpha, alpha_polyak, opt_state = update_fn(alpha, alpha_polyak, idx, features, opt_state, target_tuple)
                time_after_update = time.time()

                time_delta = time_after_update - time_before_update
                time_elapsed += time_delta
        
                if eval_cond_fn(time_elapsed, iter_counter, eval_counter):
                    # eval_metrics = eval_fn(alpha_polyak, idx, features, target_tuple)

                    # lr_to_log = get_lr(opt_state)

                    # if wandb.run is not None:
                    #     wandb.log(
                    #         {**eval_metrics, 
                    #         **{'train_step': iter_counter, 'lr': lr_to_log, 'time_elapsed': time_elapsed}})
                    # aux.append(eval_metrics)

                    eval_counter += 1
                    print("Iter: ", iter_counter, "Time: ", time_elapsed, "Eval: ", eval_counter)

                iter_counter += 1
                tqdm_update_fn(time_delta)

        self.alpha = alpha_polyak
        return self.alpha, aux

    def compute_posterior_samples( 
        self, 
        key: chex.PRNGKey, 
        n_samples: int,
        train_ds: Dataset, 
        test_ds: Dataset, 
        config: ml_collections.ConfigDict,
        use_rff: bool = True,
        n_features: int = 0,
        chol_eps: float = 1e-5,
        zero_mean: bool = True,
        metrics_list=[],
        metrics_prefix="",
        compare_exact=False):
        
        prior_covariance_key, prior_samples_key, optim_key = jr.split(key, 3)
    
        L = sampling_utils.compute_prior_covariance_factor(
                prior_covariance_key, 
                train_ds, 
                test_ds, 
                self.kernel.kernel_fn, 
                self.kernel.feature_fn,
                use_rff=use_rff, 
                n_features=n_features, 
                chol_eps=chol_eps)
        
        # Get vmapped functions for sampling from the prior and computing the posterior.
        compute_prior_samples_fn = self.get_prior_samples_fn(train_ds.N, L, use_rff)
        compute_posterior_samples_fn = self.get_posterior_samples_fn(train_ds, test_ds, zero_mean)
        compute_target_tuples_fn = optim_utils.get_target_tuples_fn(config.loss_objective)
        
        optimizer = optax.sgd(learning_rate=config.learning_rate, momentum=config.momentum, nesterov=True)
        
        grad_fn = optim_utils.get_stochastic_gradient_fn(train_ds.x, self.kernel.kernel_fn, self.noise_scale)
        update_fn = optim_utils.get_update_fn(grad_fn, optimizer, config.polyak, vmap=True)
        feature_fn = self.get_feature_fn(train_ds, config.n_features_optim, config.recompute_features)
        idx_fn = optim_utils.get_idx_fn(config.batch_size, train_ds.N, config.iterative_idx, vmap=False)

        # Call the vmapped functions
        f0_samples_train, f0_samples_test, eps0_samples = compute_prior_samples_fn(
            jr.split(prior_samples_key, n_samples))  # (n_samples, n_train), (n_samples, n_test), (n_samples, n_train)
        
        exact_samples_tuple = None
        if compare_exact:
            exact_gp = ExactGPModel(self.noise_scale, self.kernel)
            exact_gp.K = exact_gp.kernel.kernel_fn(train_ds.x, train_ds.x)
            exact_gp.compute_representer_weights(train_ds)
            
            compute_exact_alpha_samples_fn = exact_gp.get_alpha_samples_fn()
            compute_exact_posterior_samples_fn = exact_gp.get_posterior_samples_fn(train_ds, test_ds, zero_mean=False)
            compute_exact_samples_tuple_fn = eval_utils.get_exact_sample_tuples_fn(exact_gp.alpha)

            alpha_samples_exact = compute_exact_alpha_samples_fn(f0_samples_train, eps0_samples)
            posterior_samples_exact = compute_exact_posterior_samples_fn(alpha_samples_exact, f0_samples_test)

            exact_samples_tuple = compute_exact_samples_tuple_fn(
                alpha_samples_exact, posterior_samples_exact, f0_samples_test)
        
        eval_fn = eval_utils.get_eval_fn(
            metrics_list,
            train_ds,
            test_ds,
            kernel_fn=self.kernel.kernel_fn,
            feature_fn=feature_fn,
            noise_scale=self.noise_scale,
            grad_fn=grad_fn,
            metrics_prefix=metrics_prefix,
            exact_samples=exact_samples_tuple if compare_exact else None,
            vmap=True
        )

        target_tuples = compute_target_tuples_fn(f0_samples_train, eps0_samples) # (n_samples, TargetTuples)

        alphas, alphas_polyak = jnp.zeros((n_samples, train_ds.N)), jnp.zeros((n_samples, train_ds.N))
        opt_states = optimizer.init(alphas)
        
        aux = []
        for i in tqdm(range(config.iterations)):
            optim_key, idx_key, feature_key = jr.split(optim_key, 3)
            features = feature_fn(feature_key)

            idx = idx_fn(i, idx_key)

            alphas, alphas_polyak, opt_states = update_fn(alphas, alphas_polyak, idx, features, opt_states, target_tuples)

            if i % config.eval_every == 0:
                vmapped_eval_metrics = eval_fn(alphas_polyak, idx, features, target_tuples)

                aux_metrics = {}
                if "test_llh" in metrics_list or "normalised_test_llh" in metrics_list:
                    y_pred_loc = self.predictive_mean(train_ds, test_ds, recompute=False)
                    zero_mean_posterior_samples = compute_posterior_samples_fn(alphas_polyak, f0_samples_test)
                    y_pred_scale = self.predictive_variance_samples(zero_mean_posterior_samples)
                    if "test_llh" in metrics_list:
                        aux_metrics['test_llh'] = LLH(
                            test_ds.y, y_pred_loc, y_pred_scale, mu=train_ds.mu_y, sigma=train_ds.sigma_y)
                    if "normalised_test_llh" in metrics_list:
                        aux_metrics['normalised_test_llh'] = LLH(test_ds.y, y_pred_loc, y_pred_scale)

                wandb.log({**_process_vmapped_metrics(vmapped_eval_metrics),
                           **{'sample_step': i},
                           **aux_metrics})

                aux.append(vmapped_eval_metrics)

        print(f'alphas_polyak: {alphas_polyak.shape}')
        
        posterior_samples = compute_posterior_samples_fn(alphas_polyak, f0_samples_test)  # (n_samples, n_test)
        
        return posterior_samples, alphas_polyak
        
            
def _process_vmapped_metrics(vmapped_metrics):
    mean_metrics, std_metrics = {}, {}
    for k, v in vmapped_metrics.items():
        vmapped_metrics[k] = wandb.Histogram(v)
        mean_metrics[f'{k}_mean'] = jnp.mean(v)
        std_metrics[f'{k}_std'] = jnp.std(v)
        
    return {**vmapped_metrics, **mean_metrics, **std_metrics}



class CGGPModel(ExactGPModel):
    def __init__(self, noise_scale: float, kernel: Kernel, **kwargs):
        super().__init__(noise_scale=noise_scale, kernel=kernel, **kwargs)
        
        self.pivoted_chol = None
    
    def get_cg_closure_fn(self, noise_std, train_ds):
        # TODO: Pipe in the correct batch size here for KvP.
        # (K(x, x) + noise_std**2 * I) * params = y # (n_train)
        def _fn(params):
            return KvP(train_ds.x, train_ds.x, params, kernel_fn=self.kernel.kernel_fn) + params * noise_std**2

        return jax.jit(_fn)


    def get_cg_solve_fn(self, cg_closure_fn, tol, atol, maxiter, M=None, vmap=False):
        
        def _fn(v):
            return jax.scipy.sparse.linalg.cg(
                cg_closure_fn, v, tol=tol, atol=atol, maxiter=maxiter, M=M)[0]
        
        if vmap:
            return jax.jit(jax.vmap(_fn))
        else:
            return jax.jit(_fn)
    
    
    def get_cg_preconditioner_solve_fn(self, pivoted_chol):
        def _fn(v):
            """Woodbury identity-based matvec."""
            A_inv = self.noise_scale**-2
            
            U = pivoted_chol  # N, k
            V = pivoted_chol.T  #k, N
            C_inv = jnp.eye(U.shape[1]) # k, k
            # (A+U C V)^{-1} = A^{-1}- A^{-1} U (C^{-1} + V A^{-1} U )^{-1} V A^{-1}
            # (A+U C V)^{-1} v = A^{-1} v - A^{-1} U (C^{-1} + V A^{-1} U )^{-1} V A^{-1} v
            first_term = A_inv * v # (N, ) 
            
            inner_inv = (C_inv + A_inv * V @ U)  # k, k
            
            inner_solve = jax.scipy.linalg.solve(inner_inv, V @ v, assume_a='pos')  # k,
            
            second_term = (A_inv ** 2) * U @ inner_solve  # (N, )
            
            return first_term - second_term
        
        return jax.jit(_fn)
        
    
    def compute_representer_weights(
        self, 
        train_ds: Dataset, 
        test_ds: Dataset,
        config: ml_collections.ConfigDict,
        metrics_list: List[str],
        metrics_prefix: str="",
        exact_metrics: Optional[ExactPredictionsTuple] = None,
        eval_intermediate_cg: bool=False) -> Array:
        """Compute representer weights alpha by solving a linear system using Conjugate Gradients."""
        
        cg_closure_fn = self.get_cg_closure_fn(self.noise_scale, train_ds)
        
        eval_fn = eval_utils.get_eval_fn(
            metrics_list,
            train_ds,
            test_ds,
            self.kernel.kernel_fn,
            self.noise_scale,
            feature_fn=self.kernel.feature_fn,
            grad_fn=None,
            metrics_prefix=metrics_prefix,
            exact_metrics=exact_metrics
        )
        
        if config.preconditioner:
            if self.pivoted_chol is None:
                self.pivoted_chol = pivoted_cholesky(
                    self.kernel, 
                    train_ds.x, 
                    config.pivoted_chol_rank, 
                    config.pivoted_diag_rtol, 
                    config.pivoted_jitter)
            
            pivoted_solve_fn = self.get_cg_preconditioner_solve_fn(self.pivoted_chol)
        else:
            pivoted_solve_fn = None

        aux = []
        if not eval_intermediate_cg:
            cg_fn = self.get_cg_solve_fn(
                cg_closure_fn, tol=config.tol, atol=config.atol, maxiter=config.maxiter, M=pivoted_solve_fn)
        
            # Compute alpha
            self.alpha = cg_fn(train_ds.y)
            
            eval_metrics = eval_fn(self.alpha, config.maxiter, None, None)
            if wandb.run is not None:
                wandb.log({**eval_metrics, **{'train_step': config.maxiter}})
            aux.append(eval_metrics)
        else:
            aux = []
            # Compute metrics by running for longer maxiter
            alpha = None

            for i in tqdm(range(0, config.maxiter, config.eval_every)):
                    
                cg_fn = self.get_cg_solve_fn(
                    cg_closure_fn, tol=config.tol, atol=config.atol, maxiter=i, M=pivoted_solve_fn)
                
                alpha = cg_fn(train_ds.y)
                
                eval_metrics = eval_fn(alpha, i, None, None)

                if wandb.run is not None:
                    wandb.log({**eval_metrics, **{'train_step': i}})
                aux.append(eval_metrics)
            
                self.alpha = alpha
        
        return self.alpha


    def compute_posterior_samples( 
        self, 
        key: chex.PRNGKey, 
        n_samples: int,
        train_ds: Dataset, 
        test_ds: Dataset, 
        config: ml_collections.ConfigDict,
        use_rff: bool = True,
        n_features: int = 0,
        chol_eps: float = 1e-5,
        zero_mean: bool = True,
        metrics_list: list = [],
        metrics_prefix: str = "",
        compare_exact: bool = False,
        eval_intermediate_cg: bool = False,):
        
        prior_covariance_key, prior_samples_key, optim_key = jr.split(key, 3)
    
        L = sampling_utils.compute_prior_covariance_factor(
                prior_covariance_key, 
                train_ds, 
                test_ds, 
                self.kernel.kernel_fn, 
                self.kernel.feature_fn,
                use_rff=use_rff, 
                n_features=n_features, 
                chol_eps=chol_eps)
        
        cg_closure_fn = self.get_cg_closure_fn(self.noise_scale, train_ds)
        
        if config.preconditioner:
            if self.pivoted_chol is None:
                self.pivoted_chol = pivoted_cholesky(
                    self.kernel, 
                    train_ds.x, 
                    config.pivoted_chol_rank, 
                    config.pivoted_diag_rtol, 
                    config.pivoted_jitter)
            
            pivoted_solve_fn = self.get_cg_preconditioner_solve_fn(self.pivoted_chol)
        else:
            pivoted_solve_fn = None

        # Get vmapped functions for sampling from the prior and computing the posterior.
        compute_prior_samples_fn = self.get_prior_samples_fn(train_ds.N, L, use_rff)
        compute_posterior_samples_fn = self.get_posterior_samples_fn(train_ds, test_ds, zero_mean)
        compute_target_tuples_fn = optim_utils.get_target_tuples_fn(config.loss_objective)

        # Call the vmapped functions
        f0_samples_train, f0_samples_test, eps0_samples = compute_prior_samples_fn(
            jr.split(prior_samples_key, n_samples))  # (n_samples, n_train), (n_samples, n_test), (n_samples, n_train)
        
        exact_samples_tuple = None
        if compare_exact:
            exact_gp = ExactGPModel(self.noise_scale, self.kernel)
            exact_gp.K = exact_gp.kernel.kernel_fn(train_ds.x, train_ds.x)
            exact_gp.compute_representer_weights(train_ds)
            
            compute_exact_alpha_samples_fn = exact_gp.get_alpha_samples_fn()
            compute_exact_posterior_samples_fn = exact_gp.get_posterior_samples_fn(train_ds, test_ds, zero_mean=False)
            compute_exact_samples_tuple_fn = eval_utils.get_exact_sample_tuples_fn(exact_gp.alpha)

            alpha_samples_exact = compute_exact_alpha_samples_fn(f0_samples_train, eps0_samples)
            posterior_samples_exact = compute_exact_posterior_samples_fn(alpha_samples_exact, f0_samples_test)

            exact_samples_tuple = compute_exact_samples_tuple_fn(
                alpha_samples_exact, posterior_samples_exact, f0_samples_test)
        
        eval_fn = eval_utils.get_eval_fn(
            metrics_list,
            train_ds,
            test_ds,
            kernel_fn=self.kernel.kernel_fn,
            feature_fn=None,
            noise_scale=self.noise_scale,
            grad_fn=None,
            metrics_prefix=metrics_prefix,
            exact_samples=exact_samples_tuple if compare_exact else None,
            vmap=True
        )

        target_tuples = compute_target_tuples_fn(f0_samples_train, eps0_samples) # (n_samples, TargetTuples)
        
        aux = []
        alphas = None
        if not eval_intermediate_cg:
            cg_fn = self.get_cg_solve_fn(
                    cg_closure_fn, tol=config.tol, atol=config.atol, maxiter=config.maxiter, M=pivoted_solve_fn, vmap=True)
            
            alphas = cg_fn(f0_samples_train + eps0_samples)  # (n_samples, n_train)
    
            vmapped_eval_metrics = eval_fn(alphas, config.maxiter, None, target_tuples)
            
            aux_metrics = {}
            if "test_llh" in metrics_list or "normalised_test_llh" in metrics_list:
                y_pred_loc = self.predictive_mean(train_ds, test_ds, recompute=False)
                zero_mean_posterior_samples = compute_posterior_samples_fn(alphas, f0_samples_test)
                y_pred_scale = self.predictive_variance_samples(zero_mean_posterior_samples)
                if "test_llh" in metrics_list:
                    aux_metrics['test_llh'] = LLH(
                        test_ds.y, y_pred_loc, y_pred_scale, mu=train_ds.mu_y, sigma=train_ds.sigma_y)
                if "normalised_test_llh" in metrics_list:
                    aux_metrics['normalised_test_llh'] = LLH(test_ds.y, y_pred_loc, y_pred_scale)
            if wandb.run is not None:
                wandb.log({**_process_vmapped_metrics(vmapped_eval_metrics),
                            **{'sample_step': config.maxiter},
                            **aux_metrics})

            aux.append(vmapped_eval_metrics)
        else:
            
            # Number of CG iterations to evaluate.
            for i in tqdm(range(0, config.maxiter, config.eval_every)):
                
                cg_fn = self.get_cg_solve_fn(
                        cg_closure_fn, tol=config.tol, atol=config.atol, maxiter=i, M=pivoted_solve_fn, vmap=True)
                
                alphas = cg_fn(f0_samples_train + eps0_samples)  # (n_samples, n_train)
        
                vmapped_eval_metrics = eval_fn(alphas, i, None, target_tuples)

                aux_metrics = {}
                if "test_llh" in metrics_list or "normalised_test_llh" in metrics_list:
                    y_pred_loc = self.predictive_mean(train_ds, test_ds, recompute=False)
                    zero_mean_posterior_samples = compute_posterior_samples_fn(alphas, f0_samples_test)
                    y_pred_scale = self.predictive_variance_samples(zero_mean_posterior_samples)
                    if "test_llh" in metrics_list:
                        aux_metrics['test_llh'] = LLH(
                            test_ds.y, y_pred_loc, y_pred_scale, mu=train_ds.mu_y, sigma=train_ds.sigma_y)
                    if "normalised_test_llh" in metrics_list:
                        aux_metrics['normalised_test_llh'] = LLH(test_ds.y, y_pred_loc, y_pred_scale)
                if wandb.run is not None:
                    wandb.log({**_process_vmapped_metrics(vmapped_eval_metrics),
                                **{'sample_step': i},
                                **aux_metrics})

                aux.append(vmapped_eval_metrics)
        
        posterior_samples = compute_posterior_samples_fn(alphas, f0_samples_test)  # (n_samples, n_test)
        
        return posterior_samples, alphas