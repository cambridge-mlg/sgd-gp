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
    """
    Base class for kernels in Gaussian processes.
    """

    def __init__(self, kernel_config=None):
        """
        Initialize the Kernel object.

        Args:
            kernel_config (dict, optional): Configuration dictionary for the kernel. Defaults to None.
        """
        self.kernel_config = kernel_config or {}

    def kernel_fn(self, x: Array, y: Array, **kwargs) -> Array:
        """
        Compute the kernel function between two input arrays.

        Args:
            x (Array): Input array x.
            y (Array): Input array y.
            **kwargs: Additional keyword arguments.

        Returns:
            Array: Result of the kernel function.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _get_hparam(self, hparam_name: str, kwargs: Optional[dict]):
        """
        Get the value of a hyperparameter.

        Args:
            hparam_name (str): Name of the hyperparameter.
            kwargs (dict, optional): Additional keyword arguments. Defaults to None.

        Returns:
            The value of the hyperparameter.
        
        Raises:
            ValueError: If the required hyperparameter is not present in the config dict or specified in kwargs.
        """
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
        """
        Get the value of the signal scale hyperparameter.

        Args:
            kwargs (dict, optional): Additional keyword arguments. Defaults to None.

        Returns:
            The value of the signal scale hyperparameter.
        """
        return self._get_hparam("signal_scale", kwargs)

    def feature_params_fn(
        self,
        key: chex.PRNGKey,
        n_features: int,
        D: int,
        **kwargs,
    ) -> NamedTuple:
        """
        Compute the feature parameters for the kernel.

        Args:
            key (chex.PRNGKey): PRNG key.
            n_features (int): Number of features.
            D (int): Dimensionality of the input.
            **kwargs: Additional keyword arguments.

        Returns:
            NamedTuple: Result of the feature parameters computation.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def feature_fn(self, x: Array, feature_params: NamedTuple) -> Array:
        """
        Compute the features for the input array.

        Args:
            x (Array): Input array.
            feature_params (NamedTuple): Feature parameters.

        Returns:
            Array: Result of the feature computation.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class StationaryKernel(Kernel):
    """
    Represents a stationary kernel.

    Attributes:
        kernel_config (dict): Configuration parameters for the kernel.
        omega (None): Placeholder for the omega parameter.
        phi (None): Placeholder for the phi parameter.
    """

    def __init__(self, kernel_config=None):
        """
        Initializes a StationaryKernel object.

        Args:
            kernel_config (dict, optional): Configuration parameters for the kernel. Defaults to None.
        """
        self.kernel_config = kernel_config or {}
        self.omega = None
        self.phi = None

    def omega_fn(self, key: chex.PRNGKey, n_input_dims: int, n_features: int):
        """
        Computes the omega parameter.

        This method should be implemented by subclasses.

        Args:
            key (chex.PRNGKey): The random key for generating random numbers.
            n_input_dims (int): The number of input dimensions.
            n_features (int): The number of features.

        Raises:
            NotImplementedError: Subclasses should implement this method.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @partial(jax.jit, static_argnums=(0, 2))
    def phi_fn(self, key: chex.PRNGKey, n_features: int):
        """
        Computes the phi parameter.

        Args:
            key (chex.PRNGKey): The random key for generating random numbers.
            n_features (int): The number of features.

        Returns:
            Array: The computed phi parameter.
        """
        return jr.uniform(key=key, shape=(1, n_features), minval=-jnp.pi, maxval=jnp.pi)

    def _sq_dist(self, x: Array, y: Array, length_scale: Array):
        """
        Computes the squared distance between two arrays.

        Args:
            x (Array): The first array.
            y (Array): The second array.
            length_scale (Array): The length scale.

        Returns:
            Array: The computed squared distance.
        """
        x, y = x / length_scale, y / length_scale

        return jnp.sum((x[:, None] - y[None, :]) ** 2, axis=-1)

    def get_length_scale(self, kwargs: Optional[dict] = None):
        """
        Gets the length scale from the given keyword arguments.

        Args:
            kwargs (dict, optional): Optional keyword arguments. Defaults to None.

        Returns:
            Array: The computed length scale.
        """
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
        """
        Computes the feature parameters.

        Args:
            key (chex.PRNGKey): The random key for generating random numbers.
            n_features (int): The number of features.
            D (int): The dimensionality of the input.
            kwargs (dict): Optional keyword arguments.

        Returns:
            FourierFeatureParams: The computed feature parameters.
        """
        M = n_features

        signal_scale, length_scale = (
            self.get_signal_scale(kwargs),
            self.get_length_scale(kwargs),
        )

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
        """
        Computes the features.

        Args:
            x (chex.Array): The input array.
            feature_params (FourierFeatureParams): The feature parameters.

        Returns:
            chex.Array: The computed features.
        """
        return (
            feature_params.signal_scale
            * jnp.sqrt(2.0 / feature_params.M)
            * jnp.cos(
                (x / feature_params.length_scale) @ feature_params.omega
                + feature_params.phi
            )
        )


class RBFKernel(StationaryKernel):
    """
    Radial Basis Function (RBF) Kernel.

    This kernel computes the covariance between two input arrays `x` and `y`
    using the RBF kernel function. It also provides a method to generate random
    features for the kernel.

    Args:
        StationaryKernel: Base class for stationary kernels.

    Attributes:
        None

    Methods:
        kernel_fn: Computes the covariance between `x` and `y` using the RBF kernel function.
        omega_fn: Generates random features for the kernel.

    """

    @partial(jax.jit, static_argnums=(0,))
    def kernel_fn(self, x: Array, y: Array, **kwargs):
        """
        Computes the covariance between `x` and `y` using the RBF kernel function.

        Args:
            x: Input array of shape (n_samples, n_features).
            y: Input array of shape (n_samples, n_features).
            **kwargs: Additional keyword arguments.

        Returns:
            Covariance matrix of shape (n_samples, n_samples).

        """
        signal_scale, length_scale = (
            self.get_signal_scale(kwargs),
            self.get_length_scale(kwargs),
        )

        return (signal_scale**2) * jnp.exp(-0.5 * self._sq_dist(x, y, length_scale))

    @partial(jax.jit, static_argnums=(0,))
    def omega_fn(self, key: chex.PRNGKey, n_input_dims: int, n_features: int):
        """
        Generates random features for the kernel.

        Args:
            key: PRNGKey for random number generation.
            n_input_dims: Number of input dimensions.
            n_features: Number of random features to generate.

        Returns:
            Random features of shape (n_input_dims, n_features).

        """
        return jr.normal(key, shape=(n_input_dims, n_features))


class MaternKernel(StationaryKernel):
    """
    MaternKernel is a subclass of StationaryKernel that represents the Matern kernel.

    The Matern kernel is a popular choice for Gaussian process regression. It is a stationary kernel
    that is characterized by its smoothness parameter, which controls the smoothness of the resulting
    Gaussian process.

    Attributes:
        _df: The degrees of freedom parameter for the Matern kernel.
    """

    @partial(jax.jit, static_argnums=(0,))
    def kernel_fn(self, x: Array, y: Array, **kwargs):
        """
        Computes the value of the Matern kernel function for the given inputs.

        Args:
            x: The input array of shape (n_samples, n_features).
            y: The input array of shape (n_samples, n_features).
            **kwargs: Additional keyword arguments.

        Returns:
            The value of the Matern kernel function for the given inputs.
        """
        signal_scale, length_scale = (
            self.get_signal_scale(kwargs),
            self.get_length_scale(kwargs),
        )

        sq_dist = self._sq_dist(x, y, length_scale)
        sq_dist = jnp.clip(sq_dist, a_min=1e-10, a_max=None)

        dist = jnp.sqrt(sq_dist)

        normaliser = self._normaliser(dist, sq_dist)
        exponential_term = jnp.exp(-jnp.sqrt(self._df) * dist)
        return (signal_scale**2) * normaliser * exponential_term

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def omega_fn(self, key: chex.PRNGKey, n_input_dims: int, n_features: int):
        """
        Generates a random matrix from the Matern kernel.

        Args:
            key: The PRNGKey used for random number generation.
            n_input_dims: The number of input dimensions.
            n_features: The number of features.

        Returns:
            A random matrix generated from the Matern kernel.
        """
        return jr.t(key, df=self._df, shape=(n_input_dims, n_features))

    @property
    def _df(self):
        """
        Returns the degrees of freedom parameter for the Matern kernel.

        Raises:
            NotImplementedError: Subclasses should implement this method.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @staticmethod
    def _normaliser(dist: Array, sq_dist: Array):
        """
        Computes the normalizer term for the Matern kernel.

        Args:
            dist: The distance array.
            sq_dist: The squared distance array.

        Raises:
            NotImplementedError: Subclasses should implement this method.
        """
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
            jax.vmap(self._pairwise_tanimoto, in_axes=(None, 0)), in_axes=(0, None)
        )(x, y)

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
        self, x: Array, r: Array, c: Array, xi: Array, beta: Array, modulo_value: int
    ) -> Array:
        t = jnp.floor(jnp.log(x) / r + beta)  # shape D (same as input x)
        ln_y = r * (t - beta)  # also shape D
        ln_a = jnp.log(c) - ln_y - r  # also shape D

        # argmin
        a_argmin = jnp.argmin(
            ln_a
        )  # this only works for 1D inputs, vectorizing will break

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
            ),
            in_axes=(None, 0, 0, 0, 0, None),
        )(
            x,
            feature_params.r,
            feature_params.c,
            feature_params.xi,
            feature_params.beta,
            feature_params.modulo_value,
        )

        return features.T

        # Vmap over the n_features and n_train of x.


class TanimotoL1Kernel(TanimotoKernel):
    def _pairwise_tanimoto(self, x: Array, y: Array):
        return (jnp.sum(x) + jnp.sum(y) - jnp.sum(jnp.abs(x - y))) / (
            jnp.sum(x) + jnp.sum(y) + jnp.sum(jnp.abs(x - y))
        )
