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
    M: int
    r: chex.Array
    c: chex.Array
    xi: chex.Array
    modulo_value: int
    beta: chex.Array


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

    def feature_params_fn(
        self,
        key: chex.PRNGKey,
        n_features: int,
        D: int,
        **kwargs,
    ) -> NamedTuple:
        raise NotImplementedError("Subclasses should implement this method.")

    def feature_fn(self, x: Array, feature_params: NamedTuple) -> Array:
        raise NotImplementedError("Subclasses should implement this method.")


class StationaryKernel(Kernel):
    def __init__(self, kernel_config=None):
        self.kernel_config = kernel_config or {}
        self.omega = None
        self.phi = None

    def omega_fn(self, key: chex.PRNGKey, n_input_dims: int, n_features: int):
        raise NotImplementedError("Subclasses should implement this method.")

    @partial(jax.jit, static_argnums=(0, 2))
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

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def feature_params_fn(
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
    
    @partial(jax.jit, static_argnums=(0,))
    def feature_fn(self, x: chex.Array, feature_params: FourierFeatureParams):
        return (
            feature_params.signal_scale
            * jnp.sqrt(2.0 / feature_params.M)
            * jnp.cos((x / feature_params.length_scale) @ feature_params.omega + feature_params.phi)
        )


class RBFKernel(StationaryKernel):
    @partial(jax.jit, static_argnums=(0,))
    def kernel_fn(self, x: Array, y: Array, **kwargs):
        signal_scale, length_scale = self.get_signal_scale(
            kwargs
        ), self.get_length_scale(kwargs)

        return (signal_scale**2) * jnp.exp(-0.5 * self._sq_dist(x, y, length_scale))

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

    @partial(jax.jit, static_argnums=(0,))
    def kernel_fn(self, x: Array, y: Array, **kwargs) -> Array:
        r"""
        Computes the following kernel between two non-negative vectors:

        \frac{\sum_i \min(x_i, y_i)}{\sum_i \max(x_i, y_i)}

        This is just designed for scalars.
        """
        chex.assert_rank(x, 2)
        chex.assert_rank(y, 2)

        return jax.vmap(
            jax.vmap(self._pairwise_tanimoto, in_axes=(None, 0)), in_axes=(0, None))(x, y)
    
    @partial(jax.jit, static_argnums=(0, 2, 3, 4))
    def feature_params_fn(
        self,
        key: chex.PRNGKey,
        n_features: int,
        D: int,
        modulo_value: int = 8,
        **kwargs,
    ) -> TanimotoFeatureParams:
        M = n_features

        r_key_1, r_key_2, c_key_1, c_key_2, xi_key, beta_key = jr.split(key, 6)

        r = -jnp.log(jr.uniform(r_key_1, (M, D))) - jnp.log(jr.uniform(r_key_2, (M, D)))
        c = -jnp.log(jr.uniform(c_key_1, (M, D))) - jnp.log(jr.uniform(c_key_2, (M, D)))
        xi = jr.randint(xi_key, (M, D, modulo_value), 0, 2) * 2 - 1
        beta = jr.uniform(beta_key, (M, D))

        return TanimotoFeatureParams(
            M=M,
            r=r,
            c=c,
            xi=xi,
            modulo_value=modulo_value,
            beta=beta,
        )

    def _elementwise_feature_fn(
            self, x: Array, r: Array, c: Array, xi: Array, beta: Array, modulo_value: int) -> Array:
        t = jnp.floor(jnp.log(x) / r + beta)  # shape D (same as input x)
        ln_y = r * (t - beta)  # also shape D
        ln_a = jnp.log(c) - ln_y - r  # also shape D

        # argmin
        a_argmin = jnp.argmin(ln_a)  # this only works for 1D inputs, vectorizing will break

        print(a_argmin.shape, t.shape)
        t_selected = t[a_argmin].astype(jnp.int32)
        # Use this to index xi
        return xi[a_argmin, t_selected % modulo_value]

    @partial(jax.jit, static_argnums=(0,))
    def feature_fn(self, x: Array, feature_params: TanimotoFeatureParams) -> Array:
        chex.assert_rank(x, 2)

        features = jax.vmap(
            jax.vmap(
                self._elementwise_feature_fn, in_axes=(0, None, None, None, None, None)
            ), in_axes=(None, 0, 0, 0, 0, None)
        )(x, feature_params.r, feature_params.c, feature_params.xi, feature_params.beta, feature_params.modulo_value)

        return features.T

        # Vmap over the n_features and n_train of x.


class TanimotoL1Kernel(TanimotoKernel):

    def _pairwise_tanimoto(self, x: Array, y: Array):
        return (jnp.sum(x) + jnp.sum(y) - jnp.sum(jnp.abs(x - y))) / (
                jnp.sum(x) + jnp.sum(y) + jnp.sum(jnp.abs(x - y)))
