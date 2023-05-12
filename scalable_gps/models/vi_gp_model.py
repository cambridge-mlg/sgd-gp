from typing import List, Optional, Any
import gpjax as gpx
import jax.numpy as jnp
import jax
import ml_collections
from ml_collections import ConfigDict
import optax
import wandb
from chex import Array

from scalable_gps.data import Dataset
from scalable_gps.kernels import Kernel
from scalable_gps.SVGP import regression_SVGP, sample_from_qu
from scalable_gps.models.exact_gp_model import ExactGPModel
import chex
from scalable_gps.linalg_utils import KvP, solve_K_inv_v
from functools import partial


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
            ARD=self.kernel.kernel_config["use_ard"],
            noise_scale=noise_scale,
            key=key,
            inducing_init=config.vi_config.inducing_init,
        )
        # self.function_dist = None
        # self.predictive_dist = None

    def compute_representer_weights(
        self,
        key: chex.PRNGKey,
        train_ds: Dataset,
        test_ds: Dataset,
        config: ml_collections.ConfigDict,
        metrics_list: List[str],
        metrics_prefix: str = "",
        exact_metrics: Optional[Any] = None,
        recompute: Optional[bool] = None,
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
        )

        self.vi_params, loss = optimised_state.unpack()

        # (
        #     self.function_dist,
        #     self.predictive_dist,
        # ) = self.get_predictive(self.vi_params, test_ds.x)

        if wandb.run is not None:
            for loss_val in loss:
                wandb.log({"loss": loss_val})

    def predictive_mean(
        self,
        train_ds: Dataset,
        test_ds: Dataset,
        batch_size: int = 256,
        recompute: bool = True,
    ) -> Array:
        del recompute

        if self.predictive_dist is None or self.function_dist is None and not recompute:
            raise ValueError(
                "vi_params is None. Please call compute_representer_weights() first."
            )

        test_preds = []
        x_test_split = jnp.split(test_ds.x, batch_size)
        for x in x_test_split:
            (
                self.function_dist,
                self.predictive_dist,
            ) = self.get_predictive(self.vi_params, x)

            y_pred = self.predictive_dist.mean()
            test_preds.append(y_pred)

        self.y_pred = jnp.concatenate(test_preds, axis=0)

        return self.y_pred  # (N_test, 1)

    def predictive_variance(
        self,
        train_ds: Dataset,
        test_ds: Dataset,
        return_marginal_variance: bool = True,
        batch_size: int = 256,
        recompute: bool = False,
    ) -> chex.Array:
        del recompute
        """Compute the posterior variance of the test points."""
        if self.predictive_dist is None or self.function_dist is None and not recompute:
            raise ValueError(
                "vi_params is None. Please call compute_representer_weights() first."
            )

        if return_marginal_variance:
            test_preds = []
            x_test_split = jnp.split(test_ds.x, batch_size)
            for x in x_test_split:
                (
                    self.function_dist,
                    self.predictive_dist,
                ) = self.get_predictive(self.vi_params, x)

                y_var = jnp.diag(self.predictive_dist.variance())
                test_preds.append(y_var)

            variance = jnp.concatenate(test_preds, axis=0)

            return variance
        else:
            (
                self.function_dist,
                self.predictive_dist,
            ) = self.get_predictive(self.vi_params, test_ds.x)

            variance = self.predictive_dist.variance()
            return variance

    def compute_posterior_samples(self, key, num_samples):
        raise NotImplementedError(
            "compute_posterior_samples is broken in new minibatched prediction code -- speak to Javi if you really need a fix."
        )
        # posterior_samples = self.function_dist.sample(
        #     seed=key, sample_shape=(num_samples,)
        # )

        # zero_mean_posterior_samples = posterior_samples - self.predictive_dist.mean()

        # return zero_mean_posterior_samples

    # TODO: Biased: use the method that double counts the diagonal (first paragraph of page 28 of https://arxiv.org/pdf/2210.04994.pdf
    # TODO: Unbiased: use a mixture of isotropic Gaussian likelihood with each mixture component's mean being centred at a sample. Then we can compute joint likelihoods as in the "EFFICIENT κ-ADIC SAMPLING" section on page 26 of https://arxiv.org/pdf/2210.04994.pdf
    def predictive_variance_samples(
        self, zero_mean_posterior_samples: Array, return_marginal_variance: bool = True
    ) -> Array:
        """Compute MC estimate of posterior variance of the test points using zero mean samples from posterior."""
        if self.predictive_dist is None or self.function_dist is None:
            raise ValueError(
                "vi_params is None. Please call compute_representer_weights() first."
            )

        if return_marginal_variance:
            variance = jnp.mean(zero_mean_posterior_samples**2, axis=0)  # (N_test, 1)
            variance -= self.noise_scale**2
        else:
            n_samples = zero_mean_posterior_samples.shape[0]
            variance = (
                zero_mean_posterior_samples.T @ zero_mean_posterior_samples / n_samples
            )

            variance -= (self.noise_scale**2) * jnp.eye(variance.shape[0])

        return variance


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
