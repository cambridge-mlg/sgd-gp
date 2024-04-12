import time
from typing import List, Optional

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import wandb
from chex import Array
from ml_collections import ConfigDict
from tqdm import tqdm

from scalable_gps import eval_utils, optim_utils, sampling_utils
from scalable_gps.custom_cg import custom_cg
from scalable_gps.data import Dataset
from scalable_gps.eval_utils import mean_LLH
from scalable_gps.kernels import Kernel
from scalable_gps.linalg_utils import KvP, pivoted_cholesky
from scalable_gps.models.exact_gp_model import ExactGPModel
from scalable_gps.utils import (
    ExactPredictionsTuple,
    get_latest_saved_artifact,
    process_pmapped_and_vmapped_metrics,
    save_latest_artifact,
)


class CGGPModel(ExactGPModel):
    """
    A class representing a CGGP (Conjugate Gradients Gaussian Process) model.

    Args:
        noise_scale (float): The scale of the noise in the model.
        kernel (Kernel): The kernel function to use in the model.
        **kwargs: Additional keyword arguments.

    Attributes:
        pivoted_chol (None or Array): The pivoted Cholesky decomposition of the kernel matrix.

    Methods:
        get_cg_closure_fn: Returns a closure function for the CG (Conjugate Gradients) method.
        get_cg_solve_fn: Returns a function for solving a linear system using CG.
        get_cg_preconditioner_solve_fn: Returns a function for solving a linear system using CG with a preconditioner.
        compute_representer_weights: Computes the representer weights alpha by solving a linear system using CG.
        compute_posterior_samples: Computes posterior samples from the model.

    """
    def __init__(self, noise_scale: float, kernel: Kernel, **kwargs):
        super().__init__(noise_scale=noise_scale, kernel=kernel, **kwargs)

        self.pivoted_chol = None

    from jax._src.ad_checkpoint import _optimization_barrier

    def get_cg_closure_fn(self, noise_std, train_ds, batch_size):
        """
        Returns a closure function that computes the expression (K(x, x) + noise_std**2 * I) * params - y,
        where K(x, x) is the kernel matrix, params are the model parameters, and y is the target values.

        Args:
            noise_std (float): The standard deviation of the noise.
            train_ds (Dataset): The training dataset.
            batch_size (int): The batch size for computing the kernel matrix.

        Returns:
            closure_fn: A closure function that takes in the model parameters and returns the expression
                        (K(x, x) + noise_std**2 * I) * params - y.

        """
        def _fn(params):
            return (
                KvP(
                    train_ds.x,
                    train_ds.x,
                    params,
                    kernel_fn=self.kernel.kernel_fn,
                    batch_size=batch_size,
                )
                + params * noise_std**2
            )

        return jax.jit(_fn)

    def get_cg_solve_fn(self, cg_closure_fn, tol, atol, M=None, pmap_and_vmap=False):
        """
        Returns a function that performs conjugate gradient (CG) optimization.

        Args:
            cg_closure_fn: The closure function that defines the CG optimization problem.
            tol: The tolerance for convergence.
            atol: The absolute tolerance for convergence.
            M: The preconditioner matrix (optional).
            pmap_and_vmap: A flag indicating whether to use pmap and vmap for parallelization.

        Returns:
            A function that performs CG optimization.

        """
        def _fn(v, cg_state, maxiter):
            return custom_cg(
                cg_closure_fn,
                v,
                tol=tol,
                atol=atol,
                maxiter=maxiter,
                M=M,
                cg_state=cg_state,
            )

        if pmap_and_vmap:
            return jax.pmap(jax.vmap(_fn, in_axes=(0, 0, None)), in_axes=(0, 0, None))
        else:
            return jax.jit(_fn)

    def get_cg_preconditioner_solve_fn(self, pivoted_chol):
            """
            Returns a function that performs a solve operation using the CG preconditioner.

            Args:
                pivoted_chol (ndarray): The pivoted Cholesky decomposition of the matrix.

            Returns:
                callable: A function that takes a vector as input and returns the result of the solve operation.
            """
            def _fn(v):
                """Woodbury identity-based matvec."""
                A_inv = self.noise_scale**-2

                U = pivoted_chol  # N, k
                V = pivoted_chol.T  # k, N
                C_inv = jnp.eye(U.shape[1])  # k, k
                # (A+U C V)^{-1} = A^{-1}- A^{-1} U (C^{-1} + V A^{-1} U )^{-1} V A^{-1}
                # (A+U C V)^{-1} v = A^{-1} v - A^{-1} U (C^{-1} + V A^{-1} U )^{-1} V A^{-1} v
                first_term = A_inv * v  # (N, )

                inner_inv = C_inv + A_inv * V @ U  # k, k

                inner_solve = jax.scipy.linalg.solve(inner_inv, V @ v, assume_a="pos")  # k,

                second_term = (A_inv**2) * U @ inner_solve  # (N, )

                return first_term - second_term

            return jax.jit(_fn)

    def compute_representer_weights(
        self,
        key: chex.PRNGKey,
        train_ds: Dataset,
        test_ds: Dataset,
        config: ConfigDict,
        metrics_list: List[str] = [],
        metrics_prefix: str = "",
        exact_metrics: Optional[ExactPredictionsTuple] = None,
        recompute: Optional[bool] = None,
        artifact_name: Optional[str] = None,
    ) -> Array:
        """Compute representer weights alpha by solving a linear system using Conjugate Gradients.

        Args:
            key: The PRNG key.
            train_ds: The training dataset.
            test_ds: The test dataset.
            config: The configuration dictionary.
            metrics_list: The list of metrics to evaluate.
            metrics_prefix: The prefix for metric names.
            exact_metrics: The exact predictions tuple.
            recompute: Whether to recompute the representer weights.
            artifact_name: The name of the artifact.

        Returns:
            The representer weights alpha.
        """
        del recompute

        # To match the API.
        del key

        # If loss, err, reg in metrics list, delete them
        for metric in ["loss", "err", "reg"]:
            if metric in metrics_list:
                metrics_list.remove(metric)
        eval_fn = eval_utils.get_eval_fn(
            metrics_list,
            train_ds,
            test_ds,
            self.kernel.kernel_fn,
            self.noise_scale,
            grad_fn=None,
            metrics_prefix=metrics_prefix,
            exact_metrics=exact_metrics,
        )

        if config.preconditioner:
            precond_start_time = time.time()
            if self.pivoted_chol is None:
                self.pivoted_chol = pivoted_cholesky(
                    self.kernel,
                    train_ds.x,
                    config.pivoted_chol_rank,
                    config.pivoted_diag_rtol,
                    config.pivoted_jitter,
                )

            pivoted_solve_fn = self.get_cg_preconditioner_solve_fn(self.pivoted_chol)
            precond_time = time.time() - precond_start_time
        else:
            pivoted_solve_fn = None
            precond_time = 0.0

        if config.batch_size == 0:

            def partial_fn(batch_size):
                cg_closure_fn = self.get_cg_closure_fn(
                    self.noise_scale, train_ds, batch_size
                )
                cg_fn = self.get_cg_solve_fn(
                    cg_closure_fn, tol=config.tol, atol=config.atol, M=pivoted_solve_fn
                )
                alpha, cg_state = cg_fn(train_ds.y, None, 1)
                cg_fn(train_ds.y, cg_state, 2)

            config.batch_size = optim_utils.select_dynamic_batch_size(
                train_ds.N, partial_fn
            )
            print(
                f"Selected batch size: {config.batch_size}, (N = {train_ds.N}, D = {train_ds.D}, "
                f"length_scale dims: {self.kernel.get_length_scale().shape[-1]})"
            )
        assert config.batch_size > 0
        cg_closure_fn = self.get_cg_closure_fn(
            self.noise_scale, train_ds, config.batch_size
        )
        cg_fn = self.get_cg_solve_fn(
            cg_closure_fn, tol=config.tol, atol=config.atol, M=pivoted_solve_fn
        )

        aux = []
        alpha = None
        cg_state = None

        wall_clock_time = 0.0

        ########### ADD PREEMPTIBLE SAFE CKPT LOADING AND SAVING ####################
        all_save_steps = list(
            jnp.arange(0, config.maxiter, config.maxiter // 10).astype(int)
        )[1:]
        # print(f'All save steps: {all_save_steps}')
        most_recent_artifact_data = None
        if config.preempt_safe:
            most_recent_artifact_data = get_latest_saved_artifact(
                artifact_name, all_save_steps
            )
            print(f"Most recent artifact data: {most_recent_artifact_data}")

        if most_recent_artifact_data is not None:
            restart_step = most_recent_artifact_data["train_step"]
            alpha = most_recent_artifact_data["alpha"]
            cg_state = most_recent_artifact_data["cg_state"]
            wall_clock_time = most_recent_artifact_data["wall_clock_time"]

        for i in tqdm(range(0, config.maxiter, config.eval_every)):
            if most_recent_artifact_data is not None and i < restart_step:
                continue
            start_time = time.time()
            alpha, cg_state = cg_fn(train_ds.y, cg_state, i)
            alpha.block_until_ready()
            end_time = time.time()
            eval_metrics = eval_fn(alpha, i, None, None)
            wall_clock_time += end_time - start_time

            if config.preempt_safe and i in all_save_steps:
                artifact_data = {
                    "alpha": alpha,
                    "cg_state": cg_state,
                    "train_step": i,
                    "wall_clock_time": wall_clock_time,
                }
                save_artifact_name = f"{artifact_name}_{i}"
                print(f"Saving artifact at step {i}, {save_artifact_name}")
                save_latest_artifact(artifact_data, save_artifact_name)

            if wandb.run is not None:
                wandb.log(
                    {
                        **eval_metrics,
                        **{
                            "train_step": i,
                            "residual": cg_state[2].real,
                            "wall_clock_time": wall_clock_time + precond_time,
                        },
                    }
                )
            aux.append(eval_metrics)

            self.alpha = alpha

        return self.alpha, aux

    def compute_posterior_samples(
        self,
        key: chex.PRNGKey,
        n_samples: int,
        train_ds: Dataset,
        test_ds: Dataset,
        config: ConfigDict,
        n_features: int = 0,
        L: Optional[Array] = None,
        zero_mean: bool = True,
        metrics_list: list = [],
        metrics_prefix: str = "",
        compare_exact: bool = False,
    ):
        """
        Computes posterior samples using the given inputs.

        Args:
            key (chex.PRNGKey): The random key for generating samples.
            n_samples (int): The number of posterior samples to generate.
            train_ds (Dataset): The training dataset.
            test_ds (Dataset): The test dataset.
            config (ConfigDict): The configuration dictionary.
            n_features (int, optional): The number of features. Defaults to 0.
            L (Optional[Array], optional): The prior covariance factor. Defaults to None.
            zero_mean (bool, optional): Whether to use zero mean. Defaults to True.
            metrics_list (list, optional): The list of metrics to compute. Defaults to [].
            metrics_prefix (str, optional): The prefix for metric names. Defaults to "".
            compare_exact (bool, optional): Whether to compare with exact samples. Defaults to False.

        Returns:
            Tuple[Array, Array, Array]: A tuple containing the posterior samples, alphas, and w_samples.
        """
        prior_covariance_key, prior_samples_key, _ = jr.split(key, 3)

        if L is None:
            L = sampling_utils.compute_prior_covariance_factor(
                prior_covariance_key,
                train_ds,
                test_ds,
                self.kernel.feature_params_fn,
                self.kernel.feature_fn,
                n_features=n_features,
            )

        # Get vmapped functions for sampling from the prior and computing the posterior.
        compute_prior_samples_fn = self.get_prior_samples_fn(train_ds.N, L, pmap=True)
        compute_posterior_samples_fn = self.get_posterior_samples_fn(
            train_ds, test_ds, zero_mean, pmap=True
        )
        compute_target_tuples_fn = optim_utils.get_target_tuples_fn(
            config.loss_objective, pmap=True
        )

        # Call the pmapped and vmapped functions
        manual_split_size = 20

        n_devices = jax.device_count()
        assert n_samples % n_devices == 0
        n_samples_per_device = n_samples // n_devices
        pmappable_keys = jr.split(
            prior_samples_key, manual_split_size * n_samples
        ).reshape((manual_split_size, n_devices, n_samples_per_device, -1))
        # (n_devices, n_samples_per_device, n_train), (n_devices, n_samples_per_device, n_test)
        w_samples = []
        f0_samples_train = jnp.zeros(
            (n_devices, n_samples_per_device, train_ds.x.shape[0])
        )
        f0_samples_test = jnp.zeros(
            (n_devices, n_samples_per_device, test_ds.x.shape[0])
        )

        for ii, L_i in enumerate(jnp.split(L, manual_split_size, axis=1)):
            compute_prior_samples_fn = self.get_prior_samples_fn(
                train_ds.N, L_i, pmap=True
            )
            # (n_devices, n_samples_per_device, n_train), (n_devices, n_samples_per_device, n_test)
            pmappable_keys_ = pmappable_keys[ii]
            (
                f0_samples_train_,
                f0_samples_test_,
                eps0_samples,
                w_samples_,
            ) = compute_prior_samples_fn(pmappable_keys_)
            w_samples.append(w_samples_)
            f0_samples_train += f0_samples_train_
            f0_samples_test += f0_samples_test_

        del L
        w_samples = jnp.concatenate(w_samples, axis=-1)

        exact_samples_tuple = None
        if compare_exact:
            exact_gp = ExactGPModel(self.noise_scale, self.kernel)
            exact_gp.K = exact_gp.kernel.kernel_fn(train_ds.x, train_ds.x)
            exact_gp.compute_representer_weights(train_ds)

            # Reshape from (n_devices, n_samples_per_device, n_train) to (n_samples, n_train)
            f0_samples_train_reshaped = jax.device_put(
                f0_samples_train.reshape(n_samples, train_ds.N), jax.devices("cpu")[0]
            )
            eps0_samples_reshaped = jax.device_put(
                eps0_samples.reshape(n_samples, train_ds.N), jax.devices("cpu")[0]
            )
            f0_samples_test_reshaped = jax.device_put(
                f0_samples_test.reshape(n_samples, test_ds.N), jax.devices("cpu")[0]
            )

            compute_exact_alpha_samples_fn = exact_gp.get_alpha_samples_fn()
            compute_exact_posterior_samples_fn = exact_gp.get_posterior_samples_fn(
                train_ds, test_ds, zero_mean=False
            )
            compute_exact_samples_tuple_fn = eval_utils.get_exact_sample_tuples_fn(
                exact_gp.alpha
            )

            alpha_samples_exact = compute_exact_alpha_samples_fn(
                f0_samples_train_reshaped, eps0_samples_reshaped
            )
            posterior_samples_exact = compute_exact_posterior_samples_fn(
                alpha_samples_exact, f0_samples_test_reshaped
            )
            exact_samples_tuple = compute_exact_samples_tuple_fn(
                alpha_samples_exact, posterior_samples_exact, f0_samples_test_reshaped
            )

        for metric in ["loss", "err", "reg"]:
            if metric in metrics_list:
                metrics_list.remove(metric)
        eval_fn = eval_utils.get_eval_fn(
            metrics_list,
            train_ds,
            test_ds,
            kernel_fn=self.kernel.kernel_fn,
            noise_scale=self.noise_scale,
            grad_fn=None,
            metrics_prefix=metrics_prefix,
            exact_samples=exact_samples_tuple if compare_exact else None,
            vmap_and_pmap=True,
        )

        target_tuples = compute_target_tuples_fn(
            f0_samples_train, eps0_samples
        )  # (n_devices, n_samples_per_device, TargetTuples)

        if config.preconditioner:
            if self.pivoted_chol is None:
                self.pivoted_chol = pivoted_cholesky(
                    self.kernel,
                    train_ds.x,
                    config.pivoted_chol_rank,
                    config.pivoted_diag_rtol,
                    config.pivoted_jitter,
                )

            pivoted_solve_fn = self.get_cg_preconditioner_solve_fn(self.pivoted_chol)
        else:
            pivoted_solve_fn = None

        # if config.batch_size == 0:
        #     def partial_fn(batch_size):
        #         cg_closure_fn = self.get_cg_closure_fn(self.noise_scale, train_ds, batch_size)
        #         cg_fn = self.get_cg_solve_fn(cg_closure_fn, tol=config.tol, atol=config.atol, M=pivoted_solve_fn, vmap=True)
        #         alphas, cg_states = cg_fn(f0_samples_train + eps0_samples, None, 1)
        #         cg_fn(f0_samples_train + eps0_samples, cg_states, 2)
        #     config.batch_size = optim_utils.select_dynamic_batch_size(train_ds.N, partial_fn)
        #     print(f"Selected batch size: {config.batch_size}, (N = {train_ds.N}, D = {train_ds.D}, "
        #           f"length_scale dims: {self.kernel.get_length_scale().shape[-1]})")
        # assert config.batch_size > 0

        cg_closure_fn = self.get_cg_closure_fn(
            self.noise_scale, train_ds, config.batch_size
        )
        cg_fn = self.get_cg_solve_fn(
            cg_closure_fn,
            tol=config.tol,
            atol=config.atol,
            M=pivoted_solve_fn,
            pmap_and_vmap=True,
        )

        aux = []
        alphas = None
        cg_states = None

        @jax.jit
        @jax.vmap
        def get_residual(cg_state):
            return {"residual": cg_state[2].real}

        for i in tqdm(range(0, config.maxiter, config.eval_every)):
            # f0_samples_train + eps0_samples is (n_devices, n_samples_per_device, n_train)
            alphas, cg_states = cg_fn(
                f0_samples_train + eps0_samples, cg_states, i
            )  # (n_samples, n_train)

            pmapped_and_vmapped_eval_metrics = eval_fn(alphas, i, None, target_tuples)

            aux_metrics = {}
            if "test_llh" in metrics_list or "normalised_test_llh" in metrics_list:
                y_pred_loc = self.predictive_mean(train_ds, test_ds, recompute=False)
                zero_mean_posterior_samples = compute_posterior_samples_fn(
                    alphas, f0_samples_test
                )
                y_pred_variance = self.predictive_variance_samples(
                    zero_mean_posterior_samples.reshape(n_samples, test_ds.N),
                    add_likelihood_noise=True,
                )
                del zero_mean_posterior_samples

                if "test_llh" in metrics_list:
                    aux_metrics["test_llh"] = mean_LLH(
                        test_ds.y,
                        y_pred_loc,
                        y_pred_variance,
                        mu=train_ds.mu_y,
                        sigma=train_ds.sigma_y,
                    )
                if "normalised_test_llh" in metrics_list:
                    aux_metrics["normalised_test_llh"] = mean_LLH(
                        test_ds.y, y_pred_loc, y_pred_variance
                    )
                del y_pred_loc, y_pred_variance
            if wandb.run is not None:
                wandb.log(
                    {
                        **process_pmapped_and_vmapped_metrics(
                            pmapped_and_vmapped_eval_metrics
                        ),
                        **process_pmapped_and_vmapped_metrics(get_residual(cg_states)),
                        **{"sample_step": i},
                        **aux_metrics,
                    }
                )

            aux.append(pmapped_and_vmapped_eval_metrics)

        print(f"alphas: {alphas.shape}")
        posterior_samples = compute_posterior_samples_fn(
            alphas, f0_samples_test
        )  # (n_samples, n_test)

        return posterior_samples, alphas, w_samples
