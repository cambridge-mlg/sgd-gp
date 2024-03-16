import jax
import jax.numpy as jnp
import jax.random as jr
from chex import Array

from scalable_gps import sampling_utils
from scalable_gps.data import Dataset
from scalable_gps.kernels import Kernel
from scalable_gps.linalg_utils import KvP


class GPModel:
    def __init__(self, noise_scale: float, kernel: Kernel):
        self.alpha = None
        self.y_pred = None
        self.K = None
        self.kernel = kernel
        self.noise_scale = noise_scale

    def compute_representer_weights(self):
        raise NotImplementedError(
            "compute_representer_weights() must be implemented in the derived class."
        )

    def predictive_mean(
        self, train_ds: Dataset, test_ds: Dataset, recompute: bool = True
    ) -> Array:
        if self.alpha is None:
            raise ValueError(
                "alpha is None. Please call compute_representer_weights() first."
            )
        if recompute or self.y_pred is None:
            self.y_pred = KvP(
                test_ds.x, train_ds.x, self.alpha, kernel_fn=self.kernel.kernel_fn
            )

        return self.y_pred  # (N_test, 1)

    # TODO: Biased: use the method that double counts the diagonal (first paragraph of page 28 of https://arxiv.org/pdf/2210.04994.pdf
    # TODO: Unbiased: use a mixture of isotropic Gaussian likelihood with each mixture component's mean being centred at a sample. Then we can compute joint likelihoods as in the "EFFICIENT κ-ADIC SAMPLING" section on page 26 of https://arxiv.org/pdf/2210.04994.pdf
    def predictive_variance_samples(
        self,
        zero_mean_posterior_samples: Array,
        add_likelihood_noise: bool = False,
        return_marginal_variance: bool = True,
    ) -> Array:
        """Compute MC estimate of posterior variance of the test points using zero mean samples from posterior."""
        # zero_mean_posterior_samples = (N_samples, N_test)
        if return_marginal_variance:
            variance = jnp.mean(zero_mean_posterior_samples**2, axis=0)  # (N_test, 1)
        else:
            n_samples = zero_mean_posterior_samples.shape[0]
            variance = (
                zero_mean_posterior_samples.T @ zero_mean_posterior_samples / n_samples
            )

        if add_likelihood_noise:
            if return_marginal_variance:
                variance += self.noise_scale**2
            else:
                variance += self.noise_scale**2 * jnp.eye(variance.shape[0])

        return variance

    def get_prior_samples_fn(self, n_train, L, pmap: bool = False):
        """Vmap factory function for sampling from the prior."""

        # fn(keys) -> prior_samples
        def _fn(key):
            prior_sample_key, sample_key = jr.split(key)
            f0_sample_train, f0_sample_test, w_sample = sampling_utils.draw_f0_sample(
                prior_sample_key, n_train, L
            )
            eps0_sample = sampling_utils.draw_eps0_sample(
                sample_key, n_train, noise_scale=self.noise_scale
            )

            return f0_sample_train, f0_sample_test, eps0_sample, w_sample

        if pmap:
            return jax.pmap(jax.vmap(_fn))  # (n_devices, n_samples_per_device)
        else:
            return jax.jit(jax.vmap(_fn))

    def get_posterior_samples_fn(
        self, train_ds, test_ds, zero_mean: bool = True, pmap: bool = False
    ):
        """Vmap factory function for computing the zero mean posterior from sample."""

        def _fn(alpha_sample, f0_sample_test):
            zero_mean_posterior_sample = sampling_utils.compute_posterior_fn_sample(
                train_ds,
                test_ds,
                alpha_sample,
                self.alpha,
                f0_sample_test,
                self.kernel.kernel_fn,
                zero_mean=zero_mean,
            )

            return zero_mean_posterior_sample

        if pmap:
            return jax.pmap(jax.vmap(_fn))  # (n_devices, n_samples_per_device)
        else:
            return jax.jit(jax.vmap(_fn))

    def get_feature_params_fn(self, n_features: int, D: int, **kwargs):
        """Factory function that wraps feature_params_fn so that it is jittable."""

        def _fn(key):
            return self.kernel.feature_params_fn(key, n_features=n_features, D=D)

        return jax.jit(_fn)

    def get_feature_fn(self, x: Array):
        """Factory function that wraps feature_fn so that it is jittable."""

        def _fn(feature_params):
            return self.kernel.feature_fn(x, feature_params)

        return jax.jit(_fn)
