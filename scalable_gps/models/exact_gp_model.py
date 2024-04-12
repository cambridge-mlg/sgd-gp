from typing import Callable, List, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import wandb
from chex import Array
from ml_collections import ConfigDict
from tqdm import tqdm

from scalable_gps import sampling_utils
from scalable_gps.data import Dataset
from scalable_gps.eval_utils import RMSE
from scalable_gps.linalg_utils import KvP, solve_K_inv_v
from scalable_gps.linear_model import marginal_likelihood
from scalable_gps.models.base_gp_model import GPModel
from scalable_gps.utils import (
    HparamsTuple,
    get_gpu_or_cpu_device,
)


class ExactGPModel(GPModel):
    """
    A class representing an Exact Gaussian Process (GP) model.

    This class extends the `GPModel` base class and provides methods for computing
    representer weights, predictive variance, posterior samples, and performing
    maximum likelihood optimization for hyperparameter estimation.

    Attributes:
        K (Array): The Gram matrix of the training data.
        alpha (Array): The representer weights alpha.

    Methods:
        compute_representer_weights: Computes the representer weights alpha.
        predictive_variance: Computes the posterior variance of the test points.
        get_alpha_samples_fn: Returns a function that computes alpha samples.
        compute_posterior_samples: Computes posterior samples.
        get_mll_loss_fn: Returns a function that computes the negative marginal log-likelihood loss.
        get_mll_update_fn: Returns a function that performs MLL optimization update.
        compute_mll_optim: Performs maximum likelihood optimization for hyperparameter estimation.
    """
    def compute_representer_weights(
        self,
        train_ds: Dataset,
        recompute: bool = False,
        test_ds: Optional[Dataset] = None,
        config: Optional[ConfigDict] = None,
        metrics_list: Optional[List[str]] = None,
        metrics_prefix: Optional[str] = None,
        exact_metrics: Optional[List] = None,
        key: Optional[chex.PRNGKey] = None,
    ) -> Tuple[Array, Optional[HparamsTuple]]:
        """
        Compute the representer weights alpha by solving alpha = (K + sigma^2 I)^{-1} y exactly.

        Args:
            train_ds (Dataset): The training dataset.
            recompute (bool, optional): Whether to recompute the kernel matrix. Defaults to False.
            test_ds (Optional[Dataset], optional): The test dataset. Defaults to None.
            config (Optional[ConfigDict], optional): Configuration dictionary. Defaults to None.
            metrics_list (Optional[List[str]], optional): List of metrics. Defaults to None.
            metrics_prefix (Optional[str], optional): Prefix for metrics. Defaults to None.
            exact_metrics (Optional[List], optional): List of exact metrics. Defaults to None.
            key (Optional[chex.PRNGKey], optional): PRNG key. Defaults to None.

        Returns:
            Tuple[Array, Optional[HparamsTuple]]: The computed representer weights alpha and None.
        """
        del test_ds, config, metrics_list, metrics_prefix, exact_metrics, key

        # Compute Kernel exactly
        if recompute or self.K is None:
            self.K = self.kernel.kernel_fn(train_ds.x, train_ds.x)

        # Compute the representer weights by solving alpha = (K + sigma^2 I)^{-1} y
        self.alpha = solve_K_inv_v(self.K, train_ds.y, noise_scale=self.noise_scale)

        return self.alpha, None

    def predictive_variance(
        self,
        train_ds: Dataset,
        test_ds: Dataset,
        add_likelihood_noise: bool = False,
        return_marginal_variance: bool = True,
    ) -> Array:
        """
        Compute the posterior variance of the test points.

        Args:
            train_ds (Dataset): The training dataset.
            test_ds (Dataset): The test dataset.
            add_likelihood_noise (bool, optional): Whether to add likelihood noise to the variance calculation. Defaults to False.
            return_marginal_variance (bool, optional): Whether to return the marginal variance. Defaults to True.

        Returns:
            Array: The computed posterior variance.

        """
        K_test = self.kernel.kernel_fn(test_ds.x, test_ds.x)  # N_test, N_test
        K_train_test = self.kernel.kernel_fn(train_ds.x, test_ds.x)  # N_train, N_test
        # Compute Kernel exactly
        self.K = (
            self.kernel.kernel_fn(train_ds.x, train_ds.x) if self.K is None else self.K
        )

        K_inv_K_train_test = solve_K_inv_v(
            self.K, K_train_test, noise_scale=self.noise_scale
        )

        variance = K_test - K_train_test.T @ K_inv_K_train_test

        if add_likelihood_noise:
            variance += self.noise_scale**2 * jnp.eye(variance.shape[0])
        if return_marginal_variance:
            return jnp.diag(variance)  # (N_test, 1)
        return variance  # (N_test, N_test)

    def get_alpha_samples_fn(self):
        """Vmap factory function that returns a function that computes alpha samples from f0 and eps0 samples.

        Returns:
            A function that takes in `f0_sample_train` and `eps0_sample` as inputs and computes `alpha_sample`.
        """
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
        config: Optional[ConfigDict] = None,
        n_features: int = 0,
        L: Optional[Array] = None,
        zero_mean: bool = True,
        metrics_list: Optional[list] = None,
        metrics_prefix: Optional[str] = None,
    ):
        """Computes n_samples posterior samples, and returns posterior_samples along with alpha_samples.

        Args:
            key: The PRNG key.
            n_samples: The number of posterior samples to compute.
            train_ds: The training dataset.
            test_ds: The test dataset.
            config: Optional configuration dictionary.
            n_features: The number of features.
            L: Optional array representing the prior covariance factor.
            zero_mean: Boolean indicating whether to use zero mean.
            metrics_list: Optional list of metrics.
            metrics_prefix: Optional prefix for metrics.

        Returns:
            A tuple containing the posterior samples, alpha samples, and w samples.

        """
        del config, metrics_list, metrics_prefix

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
        compute_prior_samples_fn = self.get_prior_samples_fn(train_ds.N, L)
        compute_alpha_samples_fn = self.get_alpha_samples_fn()
        compute_posterior_samples_fn = self.get_posterior_samples_fn(
            train_ds, test_ds, zero_mean
        )

        # Call the vmapped functions
        (
            f0_samples_train,
            f0_samples_test,
            eps0_samples,
            w_samples,
        ) = compute_prior_samples_fn(
            jr.split(prior_samples_key, n_samples)
        )  # (n_samples, n_train), (n_samples, n_test), (n_samples, n_train)

        alpha_samples = compute_alpha_samples_fn(
            f0_samples_train, eps0_samples
        )  # (n_samples, n_train)
        posterior_samples = compute_posterior_samples_fn(
            alpha_samples, f0_samples_test
        )  # (n_samples, n_test)

        chex.assert_shape(posterior_samples, (n_samples, test_ds.N))
        chex.assert_shape(alpha_samples, (n_samples, train_ds.N))

        return posterior_samples, alpha_samples, w_samples

    def get_mll_loss_fn(
        self,
        train_ds: Dataset,
        kernel_fn: Callable,
        transform: Optional[Callable] = None,
    ):
        """Factory function that wraps mll_loss_fn so that it is jittable.

        Args:
            train_ds (Dataset): The training dataset.
            kernel_fn (Callable): The kernel function.
            transform (Optional[Callable], optional): The transformation function. Defaults to None.

        Returns:
            Callable: The jitted version of the mll_loss_fn.
        """
        def _fn(log_hparams):
            return -marginal_likelihood(
                train_ds.x,
                train_ds.y,
                kernel_fn,
                hparams_tuple=log_hparams,
                transform=transform,
            )

        return jax.jit(_fn, device=get_gpu_or_cpu_device())

    def get_mll_update_fn(self, mll_loss_fn, optimizer):
        """Factory function that wraps mll_update_fn so that it is jittable.

        Args:
            mll_loss_fn: The loss function used to compute the loss and gradients.
            optimizer: The optimizer used to update the model parameters.

        Returns:
            A jittable function that computes the loss, gradients, and updates the model parameters.

        """
        def _fn(log_hparams, opt_state):
            value, grad = jax.value_and_grad(mll_loss_fn)(log_hparams)
            updates, opt_state = optimizer.update(grad, opt_state)
            return value, optax.apply_updates(log_hparams, updates), opt_state

        return jax.jit(_fn)

    def compute_mll_optim(
        self,
        init_hparams: HparamsTuple,
        train_ds: Dataset,
        config: ConfigDict,
        test_ds,
        full_train_ds: Optional[Dataset] = None,
        transform: Optional[Callable] = None,
        perform_eval: bool = True,
    ):
        """
        Computes the maximum log-likelihood (MLL) optimization for the exact GP model.

        Args:
            init_hparams (HparamsTuple): Initial hyperparameters for the GP model.
            train_ds (Dataset): Training dataset.
            config (ConfigDict): Configuration dictionary.
            test_ds: Test dataset.
            full_train_ds (Optional[Dataset], optional): Full training dataset. Defaults to None.
            transform (Optional[Callable], optional): Transformation function for hyperparameters. Defaults to None.
            perform_eval (bool, optional): Flag to perform evaluation. Defaults to True.

        Returns:
            HparamsTuple: Final optimized hyperparameters.
        """
        log_hparams = init_hparams
        hparams = None

        optimizer = optax.adam(learning_rate=config.learning_rate)
        opt_state = optimizer.init(log_hparams)

        mll_loss_fn = self.get_mll_loss_fn(
            train_ds, self.kernel.kernel_fn, transform=transform
        )
        update_fn = self.get_mll_update_fn(mll_loss_fn, optimizer)

        iterator = tqdm(range(config.iterations))
        for i in iterator:
            loss_val, log_hparams, opt_state = update_fn(log_hparams, opt_state)

            hparams = log_hparams
            if transform is not None:
                hparams = HparamsTuple(
                    length_scale=transform(log_hparams.length_scale),
                    signal_scale=transform(log_hparams.signal_scale),
                    noise_scale=transform(log_hparams.noise_scale),
                )

            # TODO: Cleanup eval if needed.
            ############################### EVAL METRICS ##################################
            # Populate evaluation metrics etc.
            if perform_eval and (
                (i == 0)
                or ((i + 1) % config.eval_every == 0)
                or (i == (config.iterations - 1))
            ):
                eval_train_ds = full_train_ds if full_train_ds is not None else train_ds
                K = self.kernel.kernel_fn(
                    eval_train_ds.x,
                    eval_train_ds.x,
                    length_scale=hparams.length_scale,
                    signal_scale=hparams.signal_scale,
                )

                # Compute the representer weights by solving alpha = (K + sigma^2 I)^{-1} y
                alpha = solve_K_inv_v(
                    K, eval_train_ds.y, noise_scale=hparams.noise_scale
                )

                y_pred_test = KvP(
                    test_ds.x,
                    eval_train_ds.x,
                    alpha,
                    kernel_fn=self.kernel.kernel_fn,
                    length_scale=hparams.length_scale,
                    signal_scale=hparams.signal_scale,
                )

                test_rmse = RMSE(
                    test_ds.y,
                    y_pred_test,
                    mu=eval_train_ds.mu_y,
                    sigma=eval_train_ds.sigma_y,
                )

                normalised_test_rmse = RMSE(test_ds.y, y_pred_test)

                iterator.set_description(f"Loss: {loss_val:.4f}")
                eval_metrics = {
                    "mll": -loss_val / eval_train_ds.N,
                    "signal_scale": hparams.signal_scale,
                    "length_scale": hparams.length_scale,
                    "noise_scale": hparams.noise_scale,
                    "test_rmse": test_rmse,
                    "normalised_test_rmse": normalised_test_rmse,
                }

                if wandb.run is not None:
                    wandb.log({**eval_metrics, **{"mll_train_step": i}})
            #########################################################################

        print("Final hyperparameters: ", hparams)

        return hparams
