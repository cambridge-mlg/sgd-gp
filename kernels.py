import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial
from chex import Array, PRNGKey

# TODO: Implement Matern 3/2 and Laplace
# TODO: Implement RFF for Matern 3/2 - https://github.com/secondmind-labs/GPflux/blob/3833993ccf2e6e166dd02a1bdb7c9778d3385318/gpflux/layers/basis_functions/fourier_features/random.py#L71
# TODO: https://github.com/secondmind-labs/GPflux/blob/c8174ee3764a303349f78eb6b13eecb2b40fc9a7/gpflux/layers/basis_functions/fourier_features/random/base.py#L100

class Kernel:
    def __init__(self, kernel_config=None, feature_config=None):
        self.kernel_config = kernel_config or {}
        self.feature_config = feature_config or {}
        self.omega = None
        self.phi = None
        
    def check_required_hparams_in_config(self, required_hparams, config_dict):
        for hparam in required_hparams:
            if hparam not in config_dict:
                raise ValueError(f"Required hyperparameter '{hparam}' must be present in config dict")

    def K(self, x: Array, y: Array):
        raise NotImplementedError("Subclasses should implement this method.")

    def Phi(self, key: PRNGKey, n_features: int, x, recompute: bool=False):
        raise NotImplementedError("Subclasses should implement this method.")


class RBFKernel(Kernel):
    @partial(jax.jit, static_argnums=(0,))
    def K(self, x: Array, y: Array):
        
        self.check_required_hparams_in_config(['signal_scale', 'length_scale'], self.kernel_config)

        s = self.kernel_config['signal_scale']
        l = self.kernel_config['length_scale']
        d2 = jnp.sum((x[:, None] - y[None, :]) ** 2, axis=-1)
        return (s ** 2) * jnp.exp(-.5 * d2 / (l ** 2))
    
    
    def omega_fn(self, key, D, M):
        return jr.normal(key, shape=(D, M))

    def phi_fn(self, key, M):
        return jr.uniform(key, shape=(1, M), minval=-jnp.pi, maxval=jnp.pi)


    def Phi(self, key: PRNGKey, n_features: int, x: Array, recompute=False):
        self.check_required_hparams_in_config(['signal_scale', 'length_scale'], self.feature_config)

        s = self.feature_config['signal_scale']
        l = self.feature_config['length_scale']
        M = n_features
        D = x.shape[-1]

        if recompute or self.omega is None or self.phi is None:
        # compute single random Fourier feature for RBF kernel
            omega_key, phi_key = jr.split(key, 2)
            omega = self.omega_fn(omega_key, D, M)
            phi = self.phi_fn(phi_key, M)
        else:
            omega, phi = self.omega, self.phi

        return s * jnp.sqrt(2. / M) * jnp.cos(x @ (omega / l) + phi)