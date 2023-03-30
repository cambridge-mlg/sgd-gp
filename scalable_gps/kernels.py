from functools import partial

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from chex import Array
from typing import Union


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
    
    def omega_fn(self, key: chex.PRNGKey, n_input_dims: int, n_features: int):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def phi_fn(self, key: chex.PRNGKey, n_features: int):
        return jr.uniform(key, shape=(1, n_features), minval=-jnp.pi, maxval=jnp.pi)
    
    def _sq_dist(self, x: Array, y: Array, length_scale: Union[float, Array]):
        # TODO: does this if/else break jitting?
        if isinstance(length_scale, Array):
            D = x.shape[-1]
            chex.assert_shape(length_scale, (D,))
            length_scale = length_scale[None, :]
        else:
            chex.assert_scalar(length_scale)
        x, y = x / length_scale, y / length_scale
        return jnp.sum((x[:, None] - y[None, :]) ** 2, axis=-1)
    
    def Phi(self, key: chex.PRNGKey, n_features: int, x: Array, recompute: bool = False):

        self.check_required_hparams_in_config(["signal_scale", "length_scale"], self.kernel_config)

        M = n_features
        D = x.shape[-1]

        signal_scale = self.kernel_config["signal_scale"]
        length_scale = self.kernel_config["length_scale"]

        # TODO: does this if/else break jitting?
        if isinstance(length_scale, Array):
            chex.assert_shape(length_scale, (D,))
            length_scale = length_scale[:, None]
        else:
            chex.assert_scalar(length_scale)

        if recompute or self.omega is None or self.phi is None:
            # compute single random Fourier feature for RBF kernel
            omega_key, phi_key = jr.split(key, 2)
            omega = self.omega_fn(omega_key, D, M)
            phi = self.phi_fn(phi_key, M)
        else:
            omega, phi = self.omega, self.phi

        return signal_scale * jnp.sqrt(2.0 / M) * jnp.cos(x @ (omega / length_scale) + phi)


class RBFKernel(Kernel):
    @partial(jax.jit, static_argnums=(0,))
    def K(self, x: Array, y: Array):

        self.check_required_hparams_in_config(["signal_scale", "length_scale"], self.kernel_config)

        signal_scale = self.kernel_config["signal_scale"]
        length_scale = self.kernel_config["length_scale"]

        return (signal_scale**2) * jnp.exp(-0.5 * self._sq_dist(x, y, length_scale))

    def omega_fn(self, key: chex.PRNGKey, n_input_dims: int, n_features: int):
        return jr.normal(key, shape=(n_input_dims, n_features))


class MaternKernel(Kernel):
    @partial(jax.jit, static_argnums=(0,))
    def K(self, x: Array, y: Array):

        self.check_required_hparams_in_config(
            ["signal_scale", "length_scale"], self.kernel_config
        )

        signal_scale = self.kernel_config["signal_scale"]
        length_scale = self.kernel_config["length_scale"]

        sq_dist = self._sq_dist(x, y, length_scale)
        dist = jnp.sqrt(sq_dist)

        normaliser = self._normaliser(dist, sq_dist)
        exponential_term = jnp.exp(-jnp.sqrt(self._df()) * dist)
        return (signal_scale**2) * normaliser * exponential_term
    
    def omega_fn(self, key: chex.PRNGKey, n_input_dims: int, n_features: int):
        return jr.t(key, df=self._df(), shape=(n_input_dims, n_features))

    def _df(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def _normaliser(self, dist: Array, sq_dist: Array):
        raise NotImplementedError("Subclasses should implement this method.")


class Matern12Kernel(MaternKernel):
    def _df(self):
        return 1.0
    
    def _normaliser(self, dist: Array, sq_dist: Array):
        return 1.0


class Matern32Kernel(MaternKernel):
    def _df(self):
        return 3.0
    
    def _normaliser(self, dist: Array, sq_dist: Array):
        return jnp.sqrt(3.0) * dist + 1.0
    

class Matern52Kernel(MaternKernel):
    def _df(self):
        return 5.0
    
    def _normaliser(self, dist: Array, sq_dist: Array):
        return jnp.sqrt(5.0) * dist + (5.0 / 3.0) * sq_dist + 1.0 
   