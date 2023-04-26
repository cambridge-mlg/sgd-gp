import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from chex import Array, PRNGKey
from typing import Callable

from scalable_gps.kernels import Kernel
from scalable_gps.data import Dataset


def init(
        seed: int,
        D: int,
        kernel: Kernel, 
        n_features: int = 10000,
        n_data_init: int = 10000,
        minval: float = -1.0,
        maxval: float = 1.0):
    """
    Initialise thompson sampling in unit hypercube [-1, 1]^D
    by constructing the kernel and sampling a function from the GP prior
    """
    omega_key, phi_key, w_key, data_key = jr.split(jr.PRNGKey(seed), 4)

    omega = kernel.omega_fn(omega_key, D, n_features)
    phi = kernel.phi_fn(phi_key, n_features)
    
    w = jr.normal(w_key, shape=(n_features,))

    signal_scale = kernel.get_signal_scale()
    length_scale = kernel.get_length_scale()

    @jax.jit
    def feature_fn(x):
        return signal_scale * jnp.sqrt(2.0 / n_features) * jnp.cos((x / length_scale) @ omega + phi)

    @jax.jit
    def objective_fn(x):
        return feature_fn(x) @ w

    x_init = jr.uniform(data_key, shape=(n_data_init, D), minval=minval, maxval=maxval)
    y_init = objective_fn(x_init)

    ds_init = Dataset(x_init, y_init, n_data_init, D)
    return feature_fn, objective_fn, ds_init


def get_maximum(ds: Dataset):
    idx = jnp.argmax(ds.y)
    return ds.x[idx], ds.y[idx]


def add_batch(ds: Dataset, x_batch: Array, objective_fn: Callable):
    y_batch = objective_fn(x_batch)
    x = jnp.concatenate([ds.x, x_batch], axis=0)
    y = jnp.concatenate([ds.y, y_batch], axis=0)
    N = ds.N + x_batch.shape[0]
    return Dataset(x, y, N, ds.D)


def find_friends(
        key: PRNGKey,
        ds: Dataset,
        n_friends: int,
        strategy: str = 'uniform',
        minval: float = -1.0,
        maxval: float = 1.0):
    
    if strategy == 'uniform':
        friends = jr.uniform(key, shape=(n_friends, ds.D), minval=minval, maxval=maxval)
    else:
        # TODO: implement other strategies
        raise NotImplementedError(f"Strategy '{strategy}' to find friends is not implemented.")
    
    return friends


def find_homies(
        x_friends: Array,
        y_friends: Array,
        n_homies: int):
    idx = jnp.argsort(-y_friends, axis=-1)[:, :n_homies]
    return x_friends[idx] # [n_samples, n_homies, D]


def find_besties(
        x_homies: Array,
        acquisition_fn: Callable,
        n_besties: int = 1):
    y_homies = acquisition_fn(x_homies)[0]
    
    @jax.vmap
    def argsort(x, y):
        return x[jnp.argsort(-y)[:n_besties]]
    return argsort(x_homies, y_homies)

def optimise_homies(
        x_homies: Array,
        acquisition_fn: Callable,
        learning_rate: float = 1e-3,
        iterations: int = 100):
    # optimiser = LBFGS(acquisition_fn, value_and_grad=True, maxiter=maxiter, tol=tol)
    optimiser = optax.adam(learning_rate=learning_rate)
    
    @jax.jit
    def update(x, opt_state):
        value, grad = acquisition_fn(x)
        # pass -grad to maximise function
        updates, opt_state = optimiser.update(-grad, opt_state, x)
        x = optax.apply_updates(x, updates)
        return x, opt_state, value
    
    opt_state = optimiser.init(x_homies)
    trace = []
    for _ in range(iterations):
        x_homies, opt_state, y_homies = update(x_homies, opt_state)
        trace.append((x_homies, y_homies))
    return x_homies, trace


def get_acquisition_fn(ds, alpha_map, alpha_samples, w_samples, feature_fn, kernel_fn, **kernel_kwargs):

    def _fn(x, alpha_sample, w_sample):
        # x: (D,)
        # alpha_sample: (n_train,)
        # w_sample: (n_features,)
        # return: ()
        return (feature_fn(x) @ w_sample + kernel_fn(x, ds.x, **kernel_kwargs) @ (alpha_map - alpha_sample)).squeeze()

    def _partial_vmapped_value_and_grad_fn(x):
        # x: (n_samples, n_test, D)
        # alpha_samples: (n_samples, n_train)
        # w_samples: (n_samples, n_features)
        # 1. vmap over 'n_test' for x only, 2. vmap over n_samples
        # return: (n_samples, n_test)
        return jax.jit(jax.vmap(jax.vmap(jax.value_and_grad(_fn), in_axes=(0, None, None))))(x, alpha_samples, w_samples)
    
    return _partial_vmapped_value_and_grad_fn
    
