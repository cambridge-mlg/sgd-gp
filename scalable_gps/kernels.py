from functools import partial

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from chex import Array

# TODO: Implement independent lengthscale for Matern 32.


class Kernel:
    def __init__(self, kernel_config=None):
        self.kernel_config = kernel_config or {}
        self.omega = None
        self.phi = None

    def check_required_hparams_in_config(self, required_hparams, config_dict):
        for hparam in required_hparams:
            if hparam not in config_dict:
                raise ValueError(
                    f"Required hyperparameter '{hparam}' must be present in config dict"
                )

    def K(self, x: Array, y: Array):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def omega_fn(self, key: chex.PRNGKey, num_input_dims: int, num_features: int):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def phi_fn(self, key: chex.PRNGKey, num_features: int):
        return jr.uniform(key, shape=(1, num_features), minval=-jnp.pi, maxval=jnp.pi)
    
    def _sq_dist(self, x: Array, y: Array):
        return jnp.sum((x[:, None] - y[None, :]) ** 2, axis=-1)
    
    def Phi(
        self, key: chex.PRNGKey, n_features: int, x: Array, recompute: bool = False
    ):
        self.check_required_hparams_in_config(
            ["signal_scale", "length_scale"], self.kernel_config
        )

        s = self.kernel_config["signal_scale"]
        l = self.kernel_config["length_scale"]
        M = n_features
        D = x.shape[-1]

        if recompute or self.omega is None or self.phi is None:
            # compute single random Fourier feature for RBF kernel
            omega_key, phi_key = jr.split(key, 2)
            omega = self.omega_fn(omega_key, D, M)
            phi = self.phi_fn(phi_key, M)
        else:
            omega, phi = self.omega, self.phi

        return s * jnp.sqrt(2.0 / M) * jnp.cos(x @ (omega / l) + phi)


class RBFKernel(Kernel):
    @partial(jax.jit, static_argnums=(0,))
    def K(self, x: Array, y: Array):

        self.check_required_hparams_in_config(
            ["signal_scale", "length_scale"], self.kernel_config
        )

        s = self.kernel_config["signal_scale"]
        l = self.kernel_config["length_scale"]

        return (s**2) * jnp.exp(-0.5 * self._sq_dist(x, y) / (l**2))

    def omega_fn(self, key: chex.PRNGKey, num_input_dims: int, num_features: int):
        return jr.normal(key, shape=(num_input_dims, num_features))


class Matern32Kernel(Kernel):
    @partial(jax.jit, static_argnums=(0,))
    def K(self, x: Array, y: Array):

        self.check_required_hparams_in_config(
            ["signal_scale", "length_scale"], self.kernel_config
        )

        s = self.kernel_config["signal_scale"]
        l = self.kernel_config["length_scale"]

        scaled_dist = jnp.sqrt(3.) * jnp.sqrt(self._sq_dist(x, y)) / l

        normaliser = 1 + scaled_dist
        exponential_term = jnp.exp(-scaled_dist)
        return (s**2) * normaliser * exponential_term


    def omega_fn(self, key: chex.PRNGKey, num_input_dims: int, num_features: int):
        return jr.t(key, df=3., shape=(num_input_dims, num_features))
