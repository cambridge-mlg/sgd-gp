from functools import partial
from typing import Optional, NamedTuple

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from chex import Array


class FeatureParams(NamedTuple):
    M: int
    omega: chex.Array
    phi: chex.Array
    signal_scale: float
    length_scale: float


@jax.jit
def featurise(x: chex.Array, params: FeatureParams):
    return (
        params.signal_scale
        * jnp.sqrt(2.0 / params.M)
        * jnp.cos((x / params.length_scale) @ params.omega + params.phi)
    )


class Kernel:
    def __init__(self, kernel_config=None):
        self.kernel_config = kernel_config or {}
        self.omega = None
        self.phi = None

    def check_required_hparam_in_config(self, hparam_name: str, config_dict):
        if hparam_name not in config_dict:
            raise ValueError(
                f"Required hyperparameter '{hparam_name}' must be present in config dict"
            )

    def kernel_fn(self, x: Array, y: Array, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")

    def omega_fn(self, key: chex.PRNGKey, n_input_dims: int, n_features: int):
        raise NotImplementedError("Subclasses should implement this method.")

    def phi_fn(self, key: chex.PRNGKey, n_features: int):
        return jr.uniform(key=key, shape=(1, n_features), minval=-jnp.pi, maxval=jnp.pi)

    def _sq_dist(self, x: Array, y: Array, length_scale: Array):
        x, y = x / length_scale, y / length_scale

        return jnp.sum((x[:, None] - y[None, :]) ** 2, axis=-1)

    def _get_hparam(self, hparam_name: str, kwargs: Optional[dict]):
        if kwargs is None or not kwargs:
            self.check_required_hparam_in_config(hparam_name, self.kernel_config)
            hparam = self.kernel_config[hparam_name]

        else:
            self.check_required_hparam_in_config(hparam_name, self.kernel_config)
            hparam = kwargs[hparam_name]

        return hparam

    def get_signal_scale(self, kwargs: Optional[dict] = None):
        return self._get_hparam("signal_scale", kwargs)

    def get_length_scale(self, kwargs: Optional[dict] = None):
        length_scale = self._get_hparam("length_scale", kwargs)
        length_scale = length_scale[None, :]
        chex.assert_rank(length_scale, 2)

        return length_scale

    def feature_fn(
        self,
        key: chex.PRNGKey,
        n_features: int,
        x: Array,
        recompute: bool = False,
        **kwargs,
    ):
        params = self.feature_params(
            key,
            n_features,
            x,
            recompute,
            **kwargs,
        )
        return featurise(x, params)

    def feature_params(
        self,
        key: chex.PRNGKey,
        n_features: int,
        x: Array,
        recompute: bool = False,
        **kwargs,
    ):
        M = n_features
        D = x.shape[-1]

        signal_scale, length_scale = self.get_signal_scale(
            kwargs
        ), self.get_length_scale(kwargs)
        if recompute or self.omega is None or self.phi is None:
            omega_key, phi_key = jr.split(key, 2)
            self.omega = self.omega_fn(omega_key, D, M)
            self.phi = self.phi_fn(phi_key, M)
        omega, phi = self.omega, self.phi
        return FeatureParams(
            M=M,
            omega=omega,
            phi=phi,
            signal_scale=signal_scale,
            length_scale=length_scale,
        )


class RBFKernel(Kernel):
    @partial(jax.jit, static_argnums=(0,))
    def kernel_fn(self, x: Array, y: Array, **kwargs):
        signal_scale, length_scale = self.get_signal_scale(
            kwargs
        ), self.get_length_scale(kwargs)

        return (signal_scale**2) * jnp.exp(-0.5 * self._sq_dist(x, y, length_scale))

    def omega_fn(self, key: chex.PRNGKey, n_input_dims: int, n_features: int):
        return jr.normal(key, shape=(n_input_dims, n_features))


class MaternKernel(Kernel):
    @partial(jax.jit, static_argnums=(0,))
    def kernel_fn(self, x: Array, y: Array, **kwargs):
        signal_scale, length_scale = self.get_signal_scale(
            kwargs
        ), self.get_length_scale(kwargs)

        sq_dist = self._sq_dist(x, y, length_scale)
        sq_dist = jnp.clip(sq_dist, a_min=1e-10, a_max=None)

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
