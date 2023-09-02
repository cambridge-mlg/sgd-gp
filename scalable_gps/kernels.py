from functools import partial
from typing import Optional, NamedTuple

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from chex import Array


class FourierFeatureParams(NamedTuple):
    M: int
    omega: chex.Array
    phi: chex.Array
    signal_scale: float
    length_scale: float


class TanimotoFeatureParams(NamedTuple):
    pass


class Kernel:
    def __init__(self, kernel_config=None):
        self.kernel_config = kernel_config or {}

    def kernel_fn(self, x: Array, y: Array, **kwargs) -> Array:
        raise NotImplementedError("Subclasses should implement this method.")

    def _get_hparam(self, hparam_name: str, kwargs: Optional[dict]):
        try:
            hparam = kwargs[hparam_name]
            return hparam
        except KeyError:
            pass

        try:
            hparam = self.kernel_config[hparam_name]
            return hparam
        except KeyError:
            raise ValueError(
                f"Required hyperparameter '{hparam_name}' must be present in config dict or specified in kwargs"
            )
        

    def get_signal_scale(self, kwargs: Optional[dict] = None):
        return self._get_hparam("signal_scale", kwargs)

    def feature_params(
        self,
        key: chex.PRNGKey,
        n_features: int,
        D: int,
        **kwargs,
    ) -> NamedTuple:
        raise NotImplementedError("Subclasses should implement this method.")

    def featurise(self, x: Array, params: NamedTuple) -> Array:
        raise NotImplementedError("Subclasses should implement this method.")

    def feature_fn(
        self,
        key: chex.PRNGKey,
        n_features: int,
        n_input_dims: int,
        x: Array,
        **kwargs,
    ):  
        print(n_features, n_input_dims)
        params = self.feature_params(
            key,
            n_features,
            n_input_dims,
            **kwargs,
        )
        return self.featurise(x, params)
    

class StationaryKernel(Kernel):
    def __init__(self, kernel_config=None):
        self.kernel_config = kernel_config or {}
        self.omega = None
        self.phi = None

    def omega_fn(self, key: chex.PRNGKey, n_input_dims: int, n_features: int):
        raise NotImplementedError("Subclasses should implement this method.")

    def phi_fn(self, key: chex.PRNGKey, n_features: int):
        return jr.uniform(key=key, shape=(1, n_features), minval=-jnp.pi, maxval=jnp.pi)

    def _sq_dist(self, x: Array, y: Array, length_scale: Array):
        x, y = x / length_scale, y / length_scale

        return jnp.sum((x[:, None] - y[None, :]) ** 2, axis=-1)

    def get_length_scale(self, kwargs: Optional[dict] = None):
        length_scale = self._get_hparam("length_scale", kwargs)
        length_scale = length_scale[None, :]
        chex.assert_rank(length_scale, 2)

        return length_scale

    @partial(jax.jit, static_argnums=(0,))
    def featurise(self, x: chex.Array, params: FourierFeatureParams):
        return (
            params.signal_scale
            * jnp.sqrt(2.0 / params.M)
            * jnp.cos((x / params.length_scale) @ params.omega + params.phi)
        )

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def feature_params(
        self,
        key: chex.PRNGKey,
        n_features: int,
        D: int,
        **kwargs,
    ) -> FourierFeatureParams:
        M = n_features

        signal_scale, length_scale = self.get_signal_scale(
            kwargs
        ), self.get_length_scale(kwargs)

        omega_key, phi_key = jr.split(key, 2)
        omega = self.omega_fn(omega_key, D, M)
        phi = self.phi_fn(phi_key, M)
        return FourierFeatureParams(
            M=M,
            omega=omega,
            phi=phi,
            signal_scale=signal_scale,
            length_scale=length_scale,
        )


class RBFKernel(StationaryKernel):
    @partial(jax.jit, static_argnums=(0,))
    def kernel_fn(self, x: Array, y: Array, **kwargs):
        signal_scale, length_scale = self.get_signal_scale(
            kwargs
        ), self.get_length_scale(kwargs)

        return (signal_scale**2) * jnp.exp(-0.5 * self._sq_dist(x, y, length_scale))

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def omega_fn(self, key: chex.PRNGKey, n_input_dims: int, n_features: int):
        return jr.normal(key, shape=(n_input_dims, n_features))


class MaternKernel(StationaryKernel):

    @partial(jax.jit, static_argnums=(0,))
    def kernel_fn(self, x: Array, y: Array, **kwargs):
        signal_scale, length_scale = self.get_signal_scale(
            kwargs
        ), self.get_length_scale(kwargs)

        sq_dist = self._sq_dist(x, y, length_scale)
        sq_dist = jnp.clip(sq_dist, a_min=1e-10, a_max=None)

        dist = jnp.sqrt(sq_dist)

        normaliser = self._normaliser(dist, sq_dist)
        exponential_term = jnp.exp(-jnp.sqrt(self._df) * dist)
        return (signal_scale**2) * normaliser * exponential_term

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def omega_fn(self, key: chex.PRNGKey, n_input_dims: int, n_features: int):
        return jr.t(key, df=self._df, shape=(n_input_dims, n_features))

    @property
    def _df(self):
        raise NotImplementedError("Subclasses should implement this method.")

    @staticmethod
    def _normaliser(dist: Array, sq_dist: Array):
        raise NotImplementedError("Subclasses should implement this method.")


class Matern12Kernel(MaternKernel):

    @property
    def _df(self):
        return 1.0

    @staticmethod
    def _normaliser(dist: Array, sq_dist: Array):
        return 1.0


class Matern32Kernel(MaternKernel):

    @property
    def _df(self):
        return 3.0

    @staticmethod
    def _normaliser(dist: Array, sq_dist: Array):
        return jnp.sqrt(3.0) * dist + 1.0


class Matern52Kernel(MaternKernel):

    @property
    def _df(self):
        return 5.0

    @staticmethod
    def _normaliser(dist: Array, sq_dist: Array):
        return jnp.sqrt(5.0) * dist + (5.0 / 3.0) * sq_dist + 1.0


class TanimotoKernel(Kernel):

    def _pairwise_tanimoto(self, x: Array, y: Array):
        return jnp.sum(jnp.minimum(x, y), axis=-1) / jnp.sum(jnp.maximum(x, y), axis=-1)

    def kernel_fn(self, x: Array, y: Array, **kwargs) -> Array:
        r"""
        Computes the following kernel between two non-negative vectors:

        \frac{\sum_i \min(x_i, y_i)}{\sum_i \max(x_i, y_i)}

        This is just designed for scalars.
        """
        chex.assert_rank(x, 2)
        chex.assert_rank(y, 2)

        return jax.vmap(
            jax.vmap(self._pairwise_tanimoto, in_axes=(0, None), in_axes=(None, 0))(x, y))
    
    def feature_params(
        self,
        key: chex.PRNGKey,
        n_features: int,
        D: int,
        **kwargs,
    ) -> TanimotoFeatureParams:
        raise NotImplementedError("Subclasses should implement this method.")

    def featurise(self, x: Array, params: TanimotoFeatureParams) -> Array:
        raise NotImplementedError("Subclasses should implement this method.")
