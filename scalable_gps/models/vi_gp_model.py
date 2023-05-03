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

        negative_elbo, init_state, D, self.get_predictive = self.regression_fn(
            train_ds, key
        )

        optimised_state = gpx.fit_batches(
            objective=negative_elbo,
            parameter_state=init_state,
            train_data=D,
            optax_optim=optimizer,
            num_iters=config.iterations,
            key=key,
            batch_size=config.batch_size,
        )

        self.vi_params, loss = optimised_state.unpack()

        (
            self.function_dist,
            self.predictive_dist,
        ) = self.get_predictive(self.vi_params, test_ds.x)

        if wandb.run is not None:
            for loss_val in loss:
                wandb.log({"loss": loss_val})

    def predictive_mean(
        self, train_ds: Dataset, test_ds: Dataset, recompute: bool = True
    ) -> Array:
        if self.predictive_dist is None or self.function_dist is None and not recompute:
            raise ValueError(
                "vi_params is None. Please call compute_representer_weights() first."
            )

        if recompute:
            (
                self.function_dist,
                self.predictive_dist,
            ) = self.get_predictive(self.vi_params, test_ds.x)

        self.y_pred = self.predictive_dist.mean()

        return self.y_pred  # (N_test, 1)

    def predictive_variance(
        self,
        train_ds: Dataset,
        test_ds: Dataset,
        return_marginal_variance: bool = True,
        recompute: bool = False,
    ) -> Array:
        """Compute the posterior variance of the test points."""
        if self.predictive_dist is None or self.function_dist is None and not recompute:
            raise ValueError(
                "vi_params is None. Please call compute_representer_weights() first."
            )

        if recompute:
            (
                self.function_dist,
                self.predictive_dist,
            ) = self.get_predictive(self.vi_params, test_ds.x)

        variance = self.predictive_dist.variance()

        # TODO: is this correct?
        if return_marginal_variance:
            return jnp.diag(variance)
        else:
            return variance

    def compute_posterior_samples(self, key, num_samples):
        posterior_samples = self.function_dist.sample(
            seed=key, sample_shape=(num_samples,)
        )

        zero_mean_posterior_samples = posterior_samples - self.predictive_dist.mean()

        return zero_mean_posterior_samples

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

        self.exact_gp = ExactGPModel(noise_scale=self.noise_scale, kernel=self.kernel)

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
            config,
            n_features,
            chol_eps,
            zero_mean,
            metrics_list,
            metrics_prefix,
            compare_exact,
        )

        if self.vi_params is None:
            raise ValueError(
                "Cannot compute posterior samples without first computing the variational parameters."
            )

        u_key, sample_key = jax.random.split(key)

        u_locations = self.vi_params["variational_family"]["inducing_inputs"]  # (M, D)
        u_samples = sample_from_qu(
            u_key, self.vi_params, n_samples
        ).T  # (num_samples, num_inducing)

        K = self.exact_gp.kernel.kernel_fn(u_locations, u_locations)
        self.exact_gp.K = K

        @jax.jit
        def solve(y):
            return jax.vmap(solve_K_inv_v, in_axes=(None, 0, None))(
                K, y, self.exact_gp.noise_scale
            )

        alpha_means = solve(u_samples)  # (num_samples, num_inducing)

        # TODO: bespoke vmap implementation
        w_sample_list = []
        alpha_sample_list = []
        for i in range(n_samples):
            train_ds = Dataset(
                x=u_locations,
                y=u_samples[i],
                N=len(u_samples[i]),
                D=u_locations.shape[1],
            )
            sample_key, _ = jax.random.split(sample_key)

            (
                _,
                zero_mean_alpha_samples,  # (1, num_inducing)
                w_samples,  #  (1, num_features)
            ) = self.exact_gp.compute_posterior_samples(
                key=sample_key,
                n_samples=1,
                train_ds=train_ds,
                test_ds=test_ds,
                config=None,
                use_rff=use_rff,
                L=L,
                zero_mean=True,
            )
            alpha_sample_list.append(zero_mean_alpha_samples)
            w_sample_list.append(w_samples)

        alpha_sample_list = jnp.concatenate(
            alpha_sample_list, axis=0
        )  # (num_samples, num_inducing)
        w_sample_list = jnp.concatenate(
            w_sample_list, axis=0
        )  # (num_samples, num_features)

        pseudo_representer_weights = -(alpha_means - alpha_sample_list)
        return None, pseudo_representer_weights, w_sample_list
