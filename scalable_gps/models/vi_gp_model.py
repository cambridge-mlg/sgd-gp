import time
from functools import partial
from typing import Any, List, Optional

import chex
import gpjax as gpx
import jax
import jax.numpy as jnp
import ml_collections
import optax
import wandb
from chex import Array
from ml_collections import ConfigDict

from scalable_gps.data import Dataset
from scalable_gps.kernels import Kernel
from scalable_gps.SVGP import regression_SVGP, sample_from_qu


class SVGPModel:
    """
    Variational Inference Gaussian Process Model.

    Args:
        noise_scale (float): The scale of the observation noise.
        kernel (Kernel): The kernel function to use.
        config (ml_collections.ConfigDict): Configuration dictionary.

    Attributes:
        alpha: The alpha parameter.
        y_pred: The predicted y values.
        K: The K matrix.
        kernel (Kernel): The kernel function.
        noise_scale (float): The scale of the observation noise.
        regression_fn (function): The regression function.

    Methods:
        reinit_get_predictive: Reinitializes the get_predictive function.
        compute_representer_weights: Computes the representer weights.
        predictive_mean: Computes the predictive mean.
        predictive_variance: Computes the predictive variance.
        compute_posterior_samples: Computes posterior samples.
        predictive_variance_samples: Computes predictive variance samples.
    """

    def __init__(
        self,
        noise_scale: float,
        kernel: Kernel,
        config: ml_collections.ConfigDict,
    ):
        self.alpha = None
        self.y_pred = None
        self.K = None
        self.kernel = kernel
        self.noise_scale = noise_scale
        self.regression_fn = lambda ds, key: regression_SVGP(
            ds,
            num_inducing=config.vi_config.num_inducing_points,
            kernel_name=config.kernel_name,
            kernel_config=self.kernel.kernel_config,
            ARD=len(self.kernel.kernel_config["length_scale"]) > 1,
            noise_scale=noise_scale,
            key=key,
            inducing_init=config.vi_config.inducing_init,
        )

    def reinit_get_predictive(self, train_ds, key):
        """
        Reinitializes the get_predictive function.

        Args:
            train_ds (Dataset): The training dataset.
            key: The random key.
        """
        init_key, _ = jax.random.split(key, 2)
        _, _, _, self.get_predictive = self.regression_fn(train_ds, init_key)

    def compute_representer_weights(
        self,
        key: chex.PRNGKey,
        train_ds: Dataset,
        test_ds: Dataset,
        config: ml_collections.ConfigDict,
        metrics_list: List[str] = [],
        metrics_prefix: str = "",
        exact_metrics: Optional[Any] = None,
        recompute: Optional[bool] = None,
        artifact_name: Optional[str] = None,
    ):
        """
        Computes the representer weights.

        Args:
            key (chex.PRNGKey): The random key.
            train_ds (Dataset): The training dataset.
            test_ds (Dataset): The test dataset.
            config (ml_collections.ConfigDict): Configuration dictionary.
            metrics_list (List[str], optional): List of metrics. Defaults to [].
            metrics_prefix (str, optional): Metrics prefix. Defaults to "".
            exact_metrics (Optional[Any], optional): Exact metrics. Defaults to None.
            recompute (Optional[bool], optional): Whether to recompute. Defaults to None.
            artifact_name (Optional[str], optional): Artifact name. Defaults to None.

        Returns:
            Tuple: The representer weights and loss.
        """
        del metrics_list, metrics_prefix, exact_metrics, recompute

        optimizer = optax.adam(learning_rate=config.learning_rate)

        init_key, fit_key = jax.random.split(key, 2)

        negative_elbo, init_state, D, self.get_predictive = self.regression_fn(
            train_ds, init_key
        )
        wall_clock_time = time.time()
        optimised_state = gpx.fit_batches(
            objective=negative_elbo,
            parameter_state=init_state,
            train_data=D,
            optax_optim=optimizer,
            num_iters=config.iterations,
            key=fit_key,
            batch_size=config.batch_size,
            verbose=False,
        )

        self.vi_params, loss = optimised_state.unpack()

        y_pred = self.predictive_mean(train_ds, test_ds)
        wall_clock_time = time.time() - wall_clock_time

        if wandb.run is not None:
            for loss_val in loss:
                wandb.log({"loss": loss_val})
            wandb.log({"wall_clock_time": wall_clock_time})
        return self.vi_params, loss

    def predictive_mean(
        self,
        train_ds: Dataset,
        test_ds: Dataset,
        batch_size: int = 256,
        recompute: bool = True,
    ) -> Array:
        """
        Computes the predictive mean.

        Args:
            train_ds (Dataset): The training dataset.
            test_ds (Dataset): The test dataset.
            batch_size (int, optional): Batch size. Defaults to 256.
            recompute (bool, optional): Whether to recompute. Defaults to True.

        Returns:
            Array: The predictive mean.
        """
        del recompute, train_ds

        test_preds = []
        x_test_split = jnp.array_split(test_ds.x, batch_size)
        for x in x_test_split:
            (
                _,
                predictive_dist,
            ) = self.get_predictive(self.vi_params, x)

            y_pred = predictive_dist.mean()
            test_preds.append(y_pred)

        self.y_pred = jnp.concatenate(test_preds, axis=0)

        return self.y_pred  # (N_test, 1)

    def predictive_variance(
        self,
        train_ds: Dataset,
        test_ds: Dataset,
        return_marginal_variance: bool = True,
        batch_size: int = 256,
        add_likelihood_noise: bool = False,
        recompute: bool = False,
    ) -> chex.Array:
        """
        Computes the predictive variance.

        Args:
            train_ds (Dataset): The training dataset.
            test_ds (Dataset): The test dataset.
            return_marginal_variance (bool, optional): Whether to return marginal variance. Defaults to True.
            batch_size (int, optional): Batch size. Defaults to 256.
            add_likelihood_noise (bool, optional): Whether to add likelihood noise. Defaults to False.
            recompute (bool, optional): Whether to recompute. Defaults to False.

        Returns:
            chex.Array: The predictive variance.
        """
        del recompute, train_ds
        """Compute the posterior variance of the test points."""

        # NOTE: THIS INCLUDES OBSERVATION NOISE
        if return_marginal_variance:
            test_preds = []
            x_test_split = jnp.array_split(test_ds.x, batch_size)
            for x in x_test_split:
                (
                    _,
                    predictive_dist,
                ) = self.get_predictive(self.vi_params, x)

                y_var = predictive_dist.variance()

                test_preds.append(y_var)

            variance = jnp.concatenate(test_preds, axis=0)
            if not add_likelihood_noise:
                variance -= self.noise_scale**2

            return variance
        else:
            (
                _,
                predictive_dist,
            ) = self.get_predictive(self.vi_params, test_ds.x)

            variance = predictive_dist.variance()
            if not add_likelihood_noise:
                variance -= self.noise_scale**2 * jnp.eye(variance.shape[0])

            return variance

    def compute_posterior_samples(self, key, train_ds, test_ds, num_samples):
        """
        Computes posterior samples.

        Args:
            key: The random key.
            train_ds (Dataset): The training dataset.
            test_ds (Dataset): The test dataset.
            num_samples: The number of samples.

        Returns:
            Array: The zero mean posterior samples.
        """
        (function_dist, predictive_dist) = self.get_predictive(
            self.vi_params, train_ds.x
        )
        posterior_samples = function_dist.sample(seed=key, sample_shape=(num_samples,))

        zero_mean_posterior_samples = posterior_samples - self.predictive_mean(
            train_ds, test_ds
        )

        return zero_mean_posterior_samples

    def predictive_variance_samples(
        self, zero_mean_posterior_samples: Array, return_marginal_variance: bool = True
    ):
        """
        Computes predictive variance samples.

        Args:
            zero_mean_posterior_samples (Array): The zero mean posterior samples.
            return_marginal_variance (bool, optional): Whether to return marginal variance. Defaults to True.

        Raises:
            NotImplementedError: This method is broken in new minibatched prediction code.

        Returns:
            None
        """
        raise NotImplementedError(
            "compute_posterior_samples is broken in new minibatched prediction code -- speak to Javi if you really need a fix."
        )


class SVGPThompsonInterface(SVGPModel):
    """
    A class representing the interface for the SVGP Thompson sampling model.

    Inherits from SVGPModel.

    Attributes:
        function_dist: The function distribution.
        predictive_dist: The predictive distribution.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.function_dist = None
        self.predictive_dist = None

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
            metrics_list=[],
            metrics_prefix="",
            compare_exact=False,
        ):
            """
            Computes posterior samples for the variational GP model.

            Args:
                key (chex.PRNGKey): The random key for generating samples.
                n_samples (int): The number of samples to generate.
                train_ds (Dataset): The training dataset.
                test_ds (Dataset): The test dataset.
                config (ConfigDict): The configuration dictionary.
                n_features (int, optional): The number of features. Defaults to 0.
                L (Optional[Array], optional): The array L. Defaults to None.
                zero_mean (bool, optional): Flag indicating whether to use zero mean. Defaults to True.
                metrics_list (list, optional): The list of metrics. Defaults to [].
                metrics_prefix (str, optional): The prefix for metrics. Defaults to "".
                compare_exact (bool, optional): Flag indicating whether to compare exact values. Defaults to False.

            Returns:
                Tuple: A tuple containing None, pseudo_representer_weights, and w_samples.
            """
            
            del (
                train_ds,
                test_ds,
                config,
                n_features,
                zero_mean,
                metrics_list,
                metrics_prefix,
                compare_exact,
            )
            # L is (num_inducing, num_features)

            if self.vi_params is None:
                raise ValueError(
                    "Cannot compute posterior samples without first computing the variational parameters."
                )

            u_key, w_sample_key = jax.random.split(key)

            u_locations = self.vi_params["variational_family"]["inducing_inputs"]  # (M, D)
            u_samples = sample_from_qu(
                u_key, self.vi_params, n_samples
            )  # (num_inducing, num_samples)

            K = self.kernel.kernel_fn(u_locations, u_locations)

            jit_solve = jax.jit(partial(jax.scipy.linalg.solve, assume_a="pos"))

            alpha_means = jit_solve(
                K + 1e-3 * jnp.eye(K.shape[0]), u_samples
            )  # (num_inducing, num_samples)

            w_samples = jax.random.normal(w_sample_key, shape=(n_samples, L.shape[1]))
            f0_samples = L @ w_samples.T  # (num_inducing, n_samples)
            alpha_samples = jit_solve(
                K + 1e-3 * jnp.eye(K.shape[0]), f0_samples
            )  # (num_inducing, num_samples)

            pseudo_representer_weights = -(
                alpha_means - alpha_samples
            ).T  # (num_samples, num_inducing)
            return None, pseudo_representer_weights, w_samples
