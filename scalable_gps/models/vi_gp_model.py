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

    
    def reinit_get_predictive(
        self, train_ds, key):
        init_key, _ = jax.random.split(key, 2)
        _, _, _, self.get_predictive = self.regression_fn(
            train_ds, init_key
        )
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
        del metrics_list, metrics_prefix, exact_metrics, recompute

        optimizer = optax.adam(learning_rate=config.learning_rate)
        absolute_clipping = config.absolute_clipping
        optimizer = optax.chain(
            optax.zero_nans(), optax.clip_by_global_norm(absolute_clipping), optimizer
        )

        init_key, fit_key = jax.random.split(key, 2)

        negative_elbo, init_state, D, self.get_predictive = self.regression_fn(
            train_ds, init_key
        )

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

        if wandb.run is not None:
            for loss_val in loss:
                wandb.log({"loss": loss_val})

        return self.vi_params, loss

    def predictive_mean(
        self,
        train_ds: Dataset,
        test_ds: Dataset,
        batch_size: int = 256,
        recompute: bool = True,
    ) -> Array:
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

    def compute_posterior_samples(self, key, num_samples):
        raise NotImplementedError(
            "compute_posterior_samples is broken in new minibatched prediction code -- speak to Javi if you really need a fix."
        )

    def predictive_variance_samples(
        self, zero_mean_posterior_samples: Array, return_marginal_variance: bool = True
    ):
        raise NotImplementedError(
            "compute_posterior_samples is broken in new minibatched prediction code -- speak to Javi if you really need a fix."
        )


class SVGPThompsonInterface(SVGPModel):
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
        use_rff: bool = True,
        n_features: int = 0,
        chol_eps: float = 1e-5,
        L: Optional[Array] = None,
        zero_mean: bool = True,
        metrics_list=[],
        metrics_prefix="",
        compare_exact=False,
    ):
        del (
            train_ds,
            test_ds,
            config,
            use_rff,
            n_features,
            chol_eps,
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

        alpha_means = jit_solve(K, u_samples)  # (num_inducing, num_samples)

        w_samples = jax.random.normal(w_sample_key, shape=(n_samples, L.shape[1]))
        f0_samples = L @ w_samples.T  # (num_inducing, n_samples)
        alpha_samples = jit_solve(K, f0_samples)  # (num_inducing, num_samples)

        pseudo_representer_weights = -(
            alpha_means - alpha_samples
        ).T  # (num_samples, num_inducing)
        return None, pseudo_representer_weights, w_samples
