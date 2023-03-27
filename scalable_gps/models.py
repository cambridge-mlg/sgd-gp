import jax.numpy as jnp
import jax.random as jr

from .data import Dataset
from .linalg_utils import KvP, solve_K_inv_v
from .linear_model import (
    draw_prior_function_sample,
    draw_prior_noise_sample,
    loss_fn,
)
from .train_utils import (
    get_eval_fn,
    get_sampling_eval_fn,
    get_stochastic_gradient_fn,
    get_update_fn,
    train,
)
from .metrics import RMSE
from chex import Array
from .kernels import Kernel
from collections import namedtuple
from typing import List, Tuple, Optional
from .utils import TargetTuple
import chex


class Model:
    def __init__(self):
        pass

    def compute_representer_weights(self):
        raise NotImplementedError(
            "compute_representer_weights() must be implemented in the derived class."
        )

    def compute_posterior_sample(self):
        raise NotImplementedError(
            "compute_posterior_sample() must be implemented in the derived class."
        )


class ExactGPModel(Model):
    def __init__(self, noise_scale: float, kernel: Kernel):
        super().__init__()

        self.noise_scale = noise_scale
        self.kernel = kernel
        self.K = None
        self.alpha = None

    def set_K(self, train_x: Array, recompute: bool = False):
        """Compute the kernel matrix K if it is None or recompute is True."""
        if self.K is None or recompute:
            self.K = self.kernel.K(train_x, train_x)

    def set_alpha(self, train_ds: Dataset, recompute: bool = False):
        """Compute the representer weights alpha if it is None or recompute is True."""
        if self.alpha is None or recompute:
            self.alpha = self.compute_representer_weights(train_ds)

    def compute_representer_weights(self, train_ds: Dataset):
        """Compute the representer weights alpha by solving alpha = (K + sigma^2 I)^{-1} y"""

        # Compute Kernel exactly
        self.set_K(train_ds.x)

        # Compute the representer weights by solving alpha = (K + sigma^2 I)^{-1} y
        self.alpha = solve_K_inv_v(
            self.K, train_ds.y, noise_scale=self.noise_scale
        ).squeeze()

        return self.alpha

    def predict(self, train_ds: Dataset, test_ds: Dataset):

        # Compute alpha if None
        self.set_alpha(train_ds)
        # Compute the representer weights by solving alpha = (K + sigma^2 I)^{-1} y

        y_pred = KvP(test_ds.x, train_ds.x, self.alpha, kernel_fn=self.kernel.K)

        return y_pred

    def calculate_test_rmse(self, train_ds: Dataset, test_ds: Dataset):

        y_pred = self.predict(train_ds, test_ds)
        test_rmse = RMSE(y_pred, test_ds.y, mu=train_ds.mu_y, sigma=train_ds.sigma_y)

        return test_rmse, y_pred

    def compute_posterior_sample(
        self,
        key,
        train_ds: Dataset,
        test_ds: Dataset,
        num_features_sample,
        use_rff_features: bool = False,
        zero_mean: bool = False,
    ):
        x = jnp.vstack((train_ds.x, test_ds.x))
        N, T = train_ds.N, test_ds.N

        prior_fn_key, prior_noise_key, feature_key = jr.split(key, 3)

        K_full, K_train = None, None

        if not use_rff_features:
            K_full = self.kernel.K(x, x)
            K_train = K_full[:N, :N]

        prior_fn_sample, L = draw_prior_function_sample(
            feature_key,
            prior_fn_key,
            num_features_sample,
            x,
            self.kernel.Phi,
            K_full,
            use_chol=not use_rff_features,
        )

        if use_rff_features:
            K_train = L[:N] @ L[:N].T

        prior_fn_sample_train = prior_fn_sample[:N]
        prior_fn_sample_test = prior_fn_sample[N:]

        # draw prior noise sample
        prior_noise_sample = draw_prior_noise_sample(
            prior_noise_key, N, noise_scale=self.noise_scale
        )

        alpha_sample = solve_K_inv_v(
            K_train,
            prior_fn_sample_train + prior_noise_sample,
            noise_scale=self.noise_scale,
        )

        if not zero_mean:
            alpha_sample = self.alpha - alpha_sample
        else:
            alpha_sample = -alpha_sample

        # Make prediction at x_test using one sample
        y_pred_sample = prior_fn_sample_test + KvP(
            test_ds.x, train_ds.x, alpha_sample, kernel_fn=self.kernel.K
        )

        return alpha_sample, y_pred_sample


class SamplingGPModel(Model):
    def __init__(self, noise_scale: float, kernel: Kernel, **kwargs):
        super().__init__()

        self.noise_scale = noise_scale
        self.kernel = kernel

        # Initialize alpha and alpha_polyak
        self.alpha = None
        self.alpha_polyak = None

        self.omega = None
        self.phi = None

    def _init_params(self, train_ds):
        self.alpha = jnp.zeros((train_ds.N,))
        self.alpha_polyak = jnp.zeros((train_ds.N,))

    def compute_representer_weights(
        self,
        train_ds: Dataset,
        test_ds: Dataset,
        train_config,
        key: chex.PRNGKey,
        metrics: List[str],
        metrics_prefix: str = "",
        compare_exact_vals: Optional[List] = None,
    ):

        # @ K (alpha + target)
        target_tuple = TargetTuple(
            train_ds.y.squeeze(), jnp.zeros_like(train_ds.y).squeeze()
        )
        grad_fn = get_stochastic_gradient_fn(
            train_ds.x,
            target_tuple,
            self.kernel.K,
            self.kernel.Phi,
            train_config.batch_size,
            train_config.num_features_optim,
            self.noise_scale,
            recompute_features=train_config.recompute_features,
        )

        # Define the gradient update function
        update_fn = get_update_fn(grad_fn, train_ds.N, train_config.polyak)

        eval_fn = get_eval_fn(
            metrics,
            train_ds,
            test_ds,
            loss_fn,
            grad_fn,
            target_tuple,
            self.kernel.K,
            self.noise_scale,
            metrics_prefix,
            compare_exact_vals,
        )

        # Initialise alpha and alpha_polyak
        self._init_params(train_ds)

        self.alpha_polyak, aux = train(
            key, train_config, update_fn, eval_fn, self.alpha, self.alpha_polyak
        )

        return self.alpha_polyak, aux

    def compute_posterior_sample(
        self,
        train_ds: Dataset,
        test_ds,
        train_config,
        loss_type,
        key,
        metrics,
        metrics_prefix="",
        compare_exact_vals=None,
        use_chol: bool = False,
    ):

        x = jnp.vstack((train_ds.x, test_ds.x))
        N, T = train_ds.N, test_ds.N

        prior_fn_key, prior_noise_key, feature_key = jr.split(key, 3)

        K_full = self.kernel.K(x, x) if use_chol else None
        prior_fn_sample, L = draw_prior_function_sample(
            feature_key,
            prior_fn_key,
            train_config.num_features_sample,
            x,
            self.kernel.Phi,
            K_full,
            use_chol=use_chol,
        )

        prior_fn_sample_train = prior_fn_sample[:N]
        prior_fn_sample_test = prior_fn_sample[N:]

        # draw prior noise sample
        prior_noise_sample = draw_prior_noise_sample(
            prior_noise_key, train_ds.N, noise_scale=self.noise_scale
        )

        # Depending on the three types of losses we can compute the gradient of the loss function accordingly
        target_tuple = None
        if loss_type == 1:
            target_tuple = TargetTuple(
                prior_fn_sample_train + prior_noise_sample, jnp.zeros_like(train_ds.y).squeeze()
            )
        elif loss_type == 2:
            target_tuple = TargetTuple(prior_fn_sample_train, prior_noise_sample)
        elif loss_type == 3:
            target_tuple = TargetTuple(
                jnp.zeros_like(train_ds.y), prior_fn_sample_train + prior_noise_sample
            )

        grad_fn = get_stochastic_gradient_fn(
            train_ds.x,
            target_tuple,
            self.kernel.K,
            self.kernel.Phi,
            train_config.batch_size,
            train_config.num_features_optim,
            self.noise_scale,
            recompute_features=train_config.recompute_features,
        )

        # Define the gradient update function
        update_fn = get_update_fn(grad_fn, train_ds.N, train_config.polyak)

        # TODO: Implement eval fn that calculates test RMSE with the sample?
        eval_fn = get_sampling_eval_fn(
            metrics,
            train_ds,
            test_ds,
            prior_fn_sample_test,
            self.alpha,
            loss_fn,
            grad_fn,
            target_tuple,
            self.kernel.K,
            self.noise_scale,
            metrics_prefix,
            compare_exact_vals,
        )

        # Initialise alpha and alpha_polyak for sample
        alpha = jnp.zeros((train_ds.N,))  # TODO: Can init at MAP solution.
        alpha_polyak = jnp.zeros((train_ds.N,))

        alpha_polyak, aux = train(
            key, train_config, update_fn, eval_fn, alpha, alpha_polyak
        )

        return alpha_polyak, aux
