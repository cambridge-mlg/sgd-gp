import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from chex import Array, PRNGKey
from ml_collections import ConfigDict
from typing import Callable, NamedTuple

from scalable_gps.kernels import Kernel
from scalable_gps.data import Dataset
from scalable_gps.models.base_gp_model import GPModel


class ThompsonState(NamedTuple):
    ds: Dataset
    kernel: Kernel
    L: Array
    w: Array
    max_fn_value: float
    argmax: float


# @jax.jit
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
    data_key, feature_key, w_key,  = jr.split(key, 3)

    x_init = jr.uniform(data_key, shape=(n_init, D), minval=minval, maxval=maxval)
    L_init = kernel.feature_fn(feature_key, n_features, x_init, recompute=True)
    w = jr.normal(w_key, shape=(n_features,))
    y_init = L_init @ w

    ds_init = Dataset(x_init, y_init, n_init, D)

    idx = jnp.argmax(y_init)
    argmax, max_fn_value = x_init[idx], y_init[idx]
    
    state = ThompsonState(ds=ds_init, kernel=kernel, L=L_init, w=w,
                          max_fn_value=max_fn_value, argmax=argmax)
    return state


def update_state(state: ThompsonState, x_besties: Array):
    """ Update the current state by
        - adding 'x_besties' and corresponding objective function values to the state
        - adding 'x_besties' and 'y_besties' to data of the state
        - adding features for 'x_besties' to L
        - replacing current 'argmax' and 'max_fn_value' if a new maximum has been found
    """
    # (n_samples, n_besties, D) --> (n_samples x n_besties, D)
    x_besties = jnp.reshape(x_besties, (-1, state.ds.D))
    n_features = state.L.shape[-1]
    L_besties = state.kernel.feature_fn(None, n_features, x_besties, recompute=False)
    # evaluate objective function at 'x_besties'
    y_besties = L_besties @ state.w

    # add besties to state dataset
    x = jnp.concatenate([state.ds.x, x_besties], axis=0)
    y = jnp.concatenate([state.ds.y, y_besties], axis=0)
    N = state.ds.N + x_besties.shape[0]
    # construct updated state dataset
    ds = Dataset(x, y, N, state.ds.D)

    # add features of besties to state
    L = jnp.concatenate([state.L, L_besties], axis=0)

    # find maximum of besties
    idx = jnp.argmax(y_besties)
    argmax, max_fn_value = x_besties[idx], y_besties[idx]
    # update maximum in state if appropriate
    if max_fn_value <= state.max_fn_value:
        max_fn_value = state.max_fn_value
        argmax = state.argmax
    
    # construct and return updated state
    updated_state = ThompsonState(ds=ds, kernel=state.kernel, L=L, w=state.w,
                                  max_fn_value=max_fn_value, argmax=argmax)
    return updated_state


def get_step_fn(config: ConfigDict, model: GPModel):

    if config.model_name == 'RandomSearch':

        def _fn(key: PRNGKey, state: ThompsonState):
            x_besties = find_friends(key, state, config.n_samples, method='uniform')
            return update_state(state, x_besties)
    else:
        
        # TODO: implement shared API for GPModels to make this work with any GP model
        def _fn(key: PRNGKey, state: ThompsonState):
            alpha_map = model.compute_representer_weights(state.ds, recompute=True)

            key, friends_key, samples_key = jr.split(key, 3)
            ds_friends = find_friends(friends_key, state, config.n_friends, method=config.find_friends_method)
            n_features = state.L.shape[-1]
            L_friends = state.kernel.feature_fn(None, n_features, ds_friends.x, recompute=False)
            L = jnp.concatenate([state.L, L_friends], axis=0)

            ds_friends.y, alpha_samples, w_samples =\
                model.compute_posterior_samples(samples_key, config.n_samples, state.ds, ds_friends,
                                                use_rff=True, L=L, zero_mean=False)
            
            acquisition_fn, acquisition_grad = get_acquisition_fn(state, alpha_map, alpha_samples, w_samples)
            
            x_homies = find_homies(ds_friends, config.n_homies)
            x_besties, _ = find_besties(x_homies, acquisition_fn, acquisition_grad,
                                        learning_rate=config.optim_lr,
                                        iterations=config.optim_iters,
                                        n_besties=config.n_besties,
                                        optim_trace=False)
            
            return update_state(state, x_besties)

    # TODO: make this jittable
    return _fn


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
        acquisition_grad: Callable,
        learning_rate: float = 1e-3,
        iterations: int = 100,
        n_besties: int = 1,
        optim_trace: bool = False):
    """ For every sample, independently maximise the acqusition function value of 'n_homies' exploration points.
        Return the 'n_besties' x for each sample with highest acquisition function value """
    
    optimiser = optax.adam(learning_rate=learning_rate)
    
    @jax.jit
    def update(x, opt_state):
        # pass -grad to maximise function
        updates, opt_state = optimiser.update(-acquisition_grad(x), opt_state, x)
        x = optax.apply_updates(x, updates)
        return x, opt_state
    
    opt_state = optimiser.init(x_homies)

    trace = []
    if optim_trace:
        y_homies = acquisition_fn(x_homies)
        trace.append((x_homies, acquisition_fn(x_homies)))

    for _ in range(iterations):
        x_homies, opt_state = update(x_homies, opt_state)
        if optim_trace:
            y_homies = acquisition_fn(x_homies)
            trace.append((x_homies, y_homies))
    
    @jax.vmap
    def argsort(x, y):
        return x[jnp.argsort(-y)[:n_besties]] # [n_samples, n_besties, D]
    
    if not optim_trace:
        y_homies = acquisition_fn(x_homies)
    return argsort(x_homies, y_homies), trace


def get_acquisition_fn(
        state: ThompsonState,
        alpha_map: Array, # (n_train,)
        alpha_samples: Array, # (n_samples, n_train)
        w_samples: Array, # (n_samples, n_features)
        **kernel_kwargs):
    """ Construct single acquisition function which is vmapped over samples and inputs,
        returning element-wise function values and gradients.

        acquisition_fn:   (n_samples, n_inputs, D) -> (n_samples, n_inputs)
        acquisition_grad: (n_samples, n_inputs, D) -> (n_samples, n_inputs, D)
    """

    def _fn(x, alpha_sample, w_sample):
        # x: (D,)
        # alpha_sample: (n_train,)
        # w_sample: (n_features,)
        # return: ()
        n_features = state.L.shape[-1]
        L = state.kernel.feature_fn(None, n_features, x, recompute=False)
        K = state.kernel.kernel_fn(x, state.ds.x, **kernel_kwargs)
        return (L @ w_sample + K @ (alpha_map - alpha_sample)).squeeze()

    def acquisition_fn(x):
        return jax.jit(jax.vmap(jax.vmap(_fn, in_axes=(0, None, None))))(x, alpha_samples, w_samples)
    
    def acquisition_grad(x):
        return jax.jit(jax.vmap(jax.vmap(jax.grad(_fn), in_axes=(0, None, None))))(x, alpha_samples, w_samples)
    
    return acquisition_fn, acquisition_grad


def grid_search(state: ThompsonState, minval=-1.0, maxval=1.0, grid_dim: int = 500):
    r = jnp.linspace(minval, maxval, num=grid_dim)
    xx, yy = jnp.meshgrid(r, r)
    grid = jnp.vstack([xx.ravel(), yy.ravel()])
    L = state.kernel.feature_fn(None, state.L.shape[-1], grid.T, recompute=False)
    fn_value = L @ state.w
    idx = jnp.argmax(fn_value)
    return fn_value[idx], grid.T[idx] # max_fn_value, argmax
