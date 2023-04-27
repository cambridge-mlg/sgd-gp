import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from chex import Array, PRNGKey
from typing import Callable, NamedTuple

from scalable_gps.kernels import Kernel
from scalable_gps.data import Dataset


class ThompsonState(NamedTuple):
    ds: Dataset
    L: Array
    max_fn_value: float
    argmax: float
    feature_fn: Callable
    objective_fn: Callable


def init(
        key: PRNGKey,
        D: int,
        kernel: Kernel, 
        n_features: int = 1000,
        n_init: int = 1000,
        minval: float = -1.0,
        maxval: float = 1.0):
    """
    Initialise thompson sampling in unit hypercube [-1, 1]^D
    by constructing the kernel and sampling a function from the GP prior
    """
    omega_key, phi_key, w_key, data_key = jr.split(key, 4)

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

    x_init = jr.uniform(data_key, shape=(n_init, D), minval=minval, maxval=maxval)
    y_init = objective_fn(x_init)
    ds_init = Dataset(x_init, y_init, n_init, D)

    L_init = feature_fn(x_init)

    idx = jnp.argmax(y_init)
    argmax, max_fn_value = x_init[idx], y_init[idx]

    state = ThompsonState(ds=ds_init,
                          L=L_init,
                          max_fn_value=max_fn_value,
                          argmax=argmax,
                          feature_fn=feature_fn,
                          objective_fn=objective_fn)
    return state


def add_features(L: Array, x: Array, feature_fn: Callable):
    return jnp.concatenate([L, feature_fn(x)], axis=0)


def update_state(state: ThompsonState, x_besties: Array):
    """ Update the current state by
        - adding 'x_besties' and corresponding objective function values to the state
        - adding 'x_besties' and 'y_besties' to data of the state
        - adding features for 'x_besties' to L
        - replacing current 'argmax' and 'max_fn_value' if a new maximum has been found
    """
    # (n_samples, n_besties, D) --> (n_samples x n_besties, D)
    x_besties = jnp.reshape(x_besties, (-1, state.ds.D))
    # evaluate objective function at 'x_besties'
    y_besties = state.objective_fn(x_besties)

    # add besties to state dataset
    x = jnp.concatenate([state.ds.x, x_besties], axis=0)
    y = jnp.concatenate([state.ds.y, y_besties], axis=0)
    N = state.ds.N + x_besties.shape[0]
    # construct updated state dataset
    ds = Dataset(x, y, N, state.ds.D)

    # add features of besties to state
    L = add_features(state.L, x_besties, state.feature_fn)

    # find maximum of besties
    idx = jnp.argmax(y_besties)
    argmax, max_fn_value = x_besties[idx], y_besties[idx]
    # update maximum in state if appropriate
    if max_fn_value <= state.max_fn_value:
        max_fn_value = state.max_fn_value
        argmax = state.argmax
    
    # construct and return updated state
    updated_state = ThompsonState(ds=ds,
                                  L=L,
                                  max_fn_value=max_fn_value,
                                  argmax=argmax,
                                  feature_fn=state.feature_fn,
                                  objective_fn=state.objective_fn)
    return updated_state


def find_friends(
        key: PRNGKey,
        state: ThompsonState,
        n_friends: int,
        method: str = 'uniform',
        minval: float = -1.0,
        maxval: float = 1.0):
    """ Given the current state, choose the next batch of exploration points. """

    if method == 'uniform':
        x_friends = jr.uniform(key, shape=(n_friends, state.ds.D), minval=minval, maxval=maxval)
    else:
        # TODO: implement other strategies
        raise NotImplementedError(f"Strategy '{method}' to find friends is not implemented.")
    ds_friends = Dataset(x_friends, None, n_friends, state.ds.D)
    return ds_friends


def find_homies(
        ds_friends: Dataset,
        n_homies: int):
    """ For every sample, find the 'n_homies' x which produce the highest y. """

    idx = jnp.argsort(-ds_friends.y, axis=-1)[:, :n_homies]
    return ds_friends.x[idx] # [n_samples, n_homies, D]


def find_besties(
        x_homies: Array,
        acquisition_fn: Callable,
        learning_rate: float = 1e-3,
        iterations: int = 100,
        optim_trace: bool = False,
        n_besties: int = 1):
    """ For every sample, independently maximise the acqusition function value of 'n_homies' exploration points.
        Return the 'n_besties' x for each sample with highest acquisition function value """
    
    optimiser = optax.adam(learning_rate=learning_rate)
    
    @jax.jit
    def update(x, opt_state):
        y, grad = acquisition_fn(x)
        # pass -grad to maximise function
        updates, opt_state = optimiser.update(-grad, opt_state, x)
        x = optax.apply_updates(x, updates)
        return x, y, opt_state
    
    opt_state = optimiser.init(x_homies)

    trace = []
    for _ in range(iterations):
        x_homies, y_homies, opt_state = update(x_homies, opt_state)
        if optim_trace:
            trace.append((x_homies, y_homies))
    
    @jax.vmap
    def argsort(x, y):
        return x[jnp.argsort(-y)[:n_besties]] # [n_samples, n_besties, D]
    
    return argsort(x_homies, y_homies), trace


def get_acquisition_fn(
        state: ThompsonState,
        alpha_map: Array, # (n_train,)
        alpha_samples: Array, # (n_samples, n_train)
        w_samples: Array, # (n_samples, n_features)
        kernel_fn: Callable,
        **kernel_kwargs):
    """ Construct single acquisition function which is vmapped over samples and inputs,
        returning element-wise function values and gradients.

        acquisition_fn: (n_samples, n_inputs, D) -> (n_samples, n_inputs), (n_samples, n_inputs, D)
    """

    def _fn(x, alpha_sample, w_sample):
        # x: (D,)
        # alpha_sample: (n_train,)
        # w_sample: (n_features,)
        # return: ()
        return (state.feature_fn(x) @ w_sample + kernel_fn(x, state.ds.x, **kernel_kwargs) @ (alpha_map - alpha_sample)).squeeze()

    def acquisition_fn(x):
        # x: (n_samples, n_inputs, D)
        # 1. vmap over 'n_inputs' for x only, 2. vmap over 'n_samples'
        # return: (n_samples, n_inputs)
        return jax.jit(jax.vmap(jax.vmap(jax.value_and_grad(_fn), in_axes=(0, None, None))))(x, alpha_samples, w_samples)
    
    return acquisition_fn
    
