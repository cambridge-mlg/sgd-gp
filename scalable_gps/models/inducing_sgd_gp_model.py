import time
from typing import List, Optional

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import wandb
from chex import Array
from ml_collections import ConfigDict
from tqdm import tqdm

from scalable_gps import eval_utils, optim_utils
from scalable_gps.data import Dataset
from scalable_gps.eval_utils import mean_LLH
from scalable_gps.kernels import Kernel
from scalable_gps.linalg_utils import KvP
from scalable_gps.models.base_gp_model import GPModel
from scalable_gps.optim_utils import get_lr, get_lr_and_schedule
from scalable_gps.utils import (
    ExactPredictionsTuple,
    TargetTuple,
    process_pmapped_and_vmapped_metrics,
)


class ISGDGPModel(GPModel):
    """
    Implementation of the Inducing SGD GP Model.

    Args:
        noise_scale (float): The scale of the noise.
        kernel (Kernel): The kernel function.
        **kwargs: Additional keyword arguments.

    Attributes:
        alpha (Array): The representer weights.
        y_pred (Array): The cached predictive mean.

    Methods:
        get_inducing_feature_fn: Factory function that wraps feature_fn so that it is jittable.
        predictive_mean: Computes the predictive mean for the given test dataset using the inducing SGD GP model.
        compute_representer_weights: Computes the representer weights alpha by solving alpha = (K + sigma^2 I)^{-1} y using SGD.
        compute_posterior_samples: Computes posterior samples using the inducing SGD GP model.

    Raises:
        ValueError: If alpha is None. Please call compute_representer_weights() first.
    """
    def __init__(self, noise_scale: float, kernel: Kernel, **kwargs):
        super().__init__(noise_scale=noise_scale, kernel=kernel, **kwargs)

    def get_inducing_feature_fn(
        self, train_ds: Dataset, n_features: int, recompute: bool
    ):
        """Factory function that wraps feature_fn so that it is jittable."""

        def _fn(key):
            params = self.kernel.feature_params_fn(
                key, n_features, train_ds.x.shape[-1]
            )
            features_x = self.kernel.feature_fn(train_ds.x, params)
            features_z = self.kernel.feature_fn(train_ds.z, params)
            return features_x, features_z

        return jax.jit(_fn)

    def predictive_mean(
            self, train_ds: Dataset, test_ds: Dataset, recompute: bool = True
        ) -> Array:
            """
            Computes the predictive mean for the given test dataset using the inducing SGD GP model.

            Args:
                train_ds (Dataset): The training dataset.
                test_ds (Dataset): The test dataset.
                recompute (bool, optional): Whether to recompute the predictive mean or use the cached value. 
                    Defaults to True.

            Returns:
                Array: The predictive mean for the test dataset. Shape: (N_test, 1)

            Raises:
                ValueError: If alpha is None. Please call compute_representer_weights() first.
            """
            if self.alpha is None:
                raise ValueError(
                    "alpha is None. Please call compute_representer_weights() first."
                )
            if recompute or self.y_pred is None:
                self.y_pred = KvP(
                    test_ds.x, train_ds.z, self.alpha, kernel_fn=self.kernel.kernel_fn
                )

            return self.y_pred  # (N_test, 1)

    def compute_representer_weights(
        self,
        key: chex.PRNGKey,
        train_ds: Dataset,
        test_ds: Dataset,
        config: ConfigDict,
        metrics_list: List[str],
        metrics_prefix: str = "",
        exact_metrics: Optional[ExactPredictionsTuple] = None,
        recompute: Optional[bool] = None,
    ):
        """
        Compute the representer weights alpha by solving alpha = (K + sigma^2 I)^{-1} y using SGD.

        Args:
            key: The PRNG key.
            train_ds: The training dataset.
            test_ds: The test dataset.
            config: The configuration dictionary.
            metrics_list: The list of metrics to evaluate.
            metrics_prefix: The prefix for metric names.
            exact_metrics: The exact predictions tuple.
            recompute: Whether to recompute the features.

        Returns:
            Tuple: The computed representer weights alpha and the evaluation metrics.

        Raises:
            NotImplementedError: If dynamic batch size is used (deprecated).

        """
        del recompute
        assert train_ds.z is not None

        target_tuple = TargetTuple(
            error_target=train_ds.y, regularizer_target=jnp.zeros_like(train_ds.y)
        )

        optimizer = get_lr_and_schedule(
            "sgd", config, config.lr_schedule_name, config.lr_schedule_config
        )

        # Define the gradient function
        grad_fn = optim_utils.get_inducing_stochastic_gradient_fn(
            train_ds.x, train_ds.z, self.kernel.kernel_fn, self.noise_scale
        )
        update_fn = optim_utils.get_update_fn(grad_fn, optimizer, config.polyak)
        feature_fn = self.get_inducing_feature_fn(
            train_ds, config.n_features_optim, config.recompute_features
        )

        eval_fn = eval_utils.get_inducing_eval_fn(
            metrics_list,
            train_ds,
            test_ds,
            self.kernel.kernel_fn,
            self.noise_scale,
            grad_fn=grad_fn,
            metrics_prefix=metrics_prefix,
            exact_metrics=exact_metrics,
        )

        # Initialise alpha and alpha_polyak
        N_inducing = len(train_ds.z)
        alpha, alpha_polyak = jnp.zeros((N_inducing,)), jnp.zeros((N_inducing,))

        opt_state = optimizer.init(alpha)

        if config.batch_size == 0:
            raise NotImplementedError("dynamic batch size deprecated")
        assert config.batch_size > 0

        idx_fn = optim_utils.get_idx_fn(
            config.batch_size, train_ds.N, config.iterative_idx, share_idx=False
        )

        # force JIT by running a single step
        # TODO: Wrap this in something we can call outside this function potentially. When we run 10 steps to calculate
        # num_iterations per budget, this will have to be called once there.
        idx_key, feature_key = jr.split(key, 2)
        features = feature_fn(feature_key)
        idx = idx_fn(0, idx_key)
        update_fn(alpha, alpha_polyak, idx, features, opt_state, target_tuple)

        wall_clock_time = 0.0
        aux = []
        for i in tqdm(range(config.iterations)):
            start_time = time.time()
            key, idx_key, feature_key = jr.split(key, 3)
            features = feature_fn(feature_key)
            idx = idx_fn(i, idx_key)

            alpha, alpha_polyak, opt_state = update_fn(
                alpha, alpha_polyak, idx, features, opt_state, target_tuple
            )
            end_time = time.time()
            wall_clock_time += end_time - start_time
            if i % config.eval_every == 0:
                eval_metrics = eval_fn(
                    alpha_polyak, idx, features[0], features[1], target_tuple
                )

                lr_to_log = get_lr(opt_state)

                if wandb.run is not None:
                    wandb.log(
                        {
                            **eval_metrics,
                            **{
                                "train_step": i,
                                "lr": lr_to_log,
                                "wall_clock_time": wall_clock_time,
                            },
                        }
                    )
                aux.append(eval_metrics)

        self.alpha = alpha_polyak
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
        metrics_list=[],
        metrics_prefix="",
    ):
        """
        Computes posterior samples using the inducing point stochastic gradient descent (SGD) Gaussian process model.

        Args:
            key (chex.PRNGKey): The random key for generating random numbers.
            n_samples (int): The number of posterior samples to compute.
            train_ds (Dataset): The training dataset.
            test_ds (Dataset): The test dataset.
            config (ConfigDict): The configuration dictionary.
            n_features (int, optional): The number of features. Defaults to 0.
            L (Optional[Array], optional): The array of inducing points. Defaults to None.
            zero_mean (bool, optional): Whether to use zero mean. Defaults to True.
            metrics_list (List, optional): The list of metrics to compute. Defaults to [].
            metrics_prefix (str, optional): The prefix for metric names. Defaults to "".

        Returns:
            Tuple[Array, Array, Array, List[Dict[str, float]]]: A tuple containing the posterior samples, alphas_polyak, w_samples, and auxiliary metrics.

        Raises:
            ValueError: If L is not None.

        """
        assert train_ds.z is not None
        prior_covariance_key, prior_samples_key, optim_key = jr.split(key, 3)

        if L is not None:
            raise ValueError("Inducing point SGD does not support")
        feature_params = self.kernel.feature_params_fn(
            key, n_features, train_ds.x.shape[-1]
        )
        features_x = self.kernel.featurise(train_ds.x, feature_params)
        features_x_test = self.kernel.featurise(test_ds.x, feature_params)
        L = jnp.concatenate([features_x, features_x_test], axis=0)
        # features_z = featurise(train_ds.z, feature_params)

        # Get vmapped functions for sampling from the prior and computing the posterior.
        # Will use this one for samples at x only, and manually compute function at z
        compute_prior_samples_fn = self.get_prior_samples_fn(train_ds.N, L, pmap=True)
        # adapted to use inducing points automatically
        compute_posterior_samples_fn = self.get_posterior_samples_fn(
            train_ds, test_ds, zero_mean, pmap=True
        )
        compute_target_tuples_fn = optim_utils.get_target_tuples_fn(
            config.loss_objective, pmap=True
        )

        optimizer = optax.sgd(
            learning_rate=config.learning_rate, momentum=config.momentum, nesterov=True
        )

        grad_fn = optim_utils.get_inducing_stochastic_gradient_fn(
            train_ds.x, train_ds.z, self.kernel.kernel_fn, self.noise_scale
        )
        update_fn = optim_utils.get_update_fn(
            grad_fn, optimizer, config.polyak, vmap_and_pmap=True
        )
        idx_fn = optim_utils.get_idx_fn(
            config.batch_size, train_ds.N, config.iterative_idx, share_idx=False
        )

        feature_fn = self.get_inducing_feature_fn(
            train_ds, config.n_features_optim, config.recompute_features
        )

        # Call the vmapped functions -- NOTE: eps0 samples need to be from Sig^-1

        # Call the pmapped and vmapped functions
        n_devices = jax.device_count()
        assert n_samples % n_devices == 0
        n_samples_per_device = n_samples // n_devices
        pmappable_keys = jr.split(prior_samples_key, n_samples).reshape(
            (n_devices, n_samples_per_device, -1)
        )
        # (n_devices, n_samples_per_device, n_train), (n_devices, n_samples_per_device, n_test)
        (
            f0_samples_train,
            f0_samples_test,
            eps0_samples,
            w_samples,
        ) = compute_prior_samples_fn(pmappable_keys)

        eval_fn = eval_utils.get_inducing_eval_fn(
            metrics_list,
            train_ds,
            test_ds,
            kernel_fn=self.kernel.kernel_fn,
            noise_scale=self.noise_scale,
            grad_fn=grad_fn,
            metrics_prefix=metrics_prefix,
            exact_samples=None,  # exact_samples_tuple if compare_exact else None,
            vmap_and_pmap=True,
        )

        target_tuples = compute_target_tuples_fn(
            f0_samples_train, eps0_samples
        )  # (n_devices, n_samples_per_device, TargetTuples)

        N_inducing = len(train_ds.z)
        alphas = jnp.zeros((n_devices, n_samples_per_device, N_inducing))
        alphas_polyak = jnp.zeros((n_devices, n_samples_per_device, N_inducing))

        opt_states = optimizer.init(alphas)

        idx_key, feature_key = jr.split(key, 2)

        # force JIT
        idx = idx_fn(0, idx_key)
        features = feature_fn(feature_key)
        update_fn(alphas, alphas_polyak, idx, features, opt_states, target_tuples)

        aux = []
        for i in tqdm(range(config.iterations)):
            optim_key, idx_key, feature_key = jr.split(optim_key, 3)
            features = feature_fn(feature_key)

            idx = idx_fn(i, idx_key)

            alphas, alphas_polyak, opt_states = update_fn(
                alphas, alphas_polyak, idx, features, opt_states, target_tuples
            )

            if i % config.eval_every == 0:
                pmapped_and_vmapped_eval_metrics = eval_fn(
                    alphas_polyak, idx, features[0], features[1], target_tuples
                )

                aux_metrics = {}
                if "test_llh" in metrics_list or "normalised_test_llh" in metrics_list:
                    y_pred_loc = self.predictive_mean(
                        train_ds, test_ds, recompute=False
                    )
                    zero_mean_posterior_samples = compute_posterior_samples_fn(
                        alphas_polyak, f0_samples_test
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
                            **{"sample_step": i},
                            **aux_metrics,
                        }
                    )

                aux.append(pmapped_and_vmapped_eval_metrics)

        # print(f"alphas_polyak: {alphas_polyak.shape}")

        posterior_samples = compute_posterior_samples_fn(
            alphas_polyak, f0_samples_test
        )  # (n_samples, n_test)

        return posterior_samples, alphas_polyak, w_samples, aux
