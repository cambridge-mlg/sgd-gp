import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from chex import Array, PRNGKey
from ml_collections import ConfigDict
from typing import Callable, NamedTuple, Optional

from scalable_gps.kernels import Kernel, FeatureParams, featurise
from scalable_gps.data import Dataset
from scalable_gps.models.base_gp_model import GPModel
import chex
from functools import partial
import wandb


class ThompsonState(NamedTuple):
    ds: Dataset
    # kernel: Kernel  # TODO: remove, we shouldnt carry around a non jittable kernel object
    # L: Array  # (N_points, n_features)
    feature_params: FeatureParams
    true_w: Array
    max_fn_value: float
    argmax: float
    noise_scale: float


# dont jit since we just run it once
def init(
    key: PRNGKey,
    D: int,
    kernel: Kernel,
    noise_scale: float,
    n_features: int = 1000,
    n_init: int = 1000,
    minval: float = -1.0,
    maxval: float = 1.0,
):
    """
    Initialise thompson sampling in unit hypercube [-1, 1]^D
    by constructing the kernel and sampling a function from the GP prior
    """
    (
        data_key,
        feature_key,
        w_key,
        noise_key,
    ) = jr.split(key, 4)

    x_init = jr.uniform(data_key, shape=(n_init, D), minval=minval, maxval=maxval)
    params = kernel.feature_params(feature_key, n_features, x_init, recompute=True)

    w = jr.normal(w_key, shape=(n_features,))
    y_init = featurise(x_init, params) @ w
    y_init = y_init + jr.normal(noise_key, shape=y_init.shape) * noise_scale

    ds_init = Dataset(x_init, y_init, n_init, D)

    idx = jnp.argmax(y_init)
    argmax, max_fn_value = x_init[idx], y_init[idx]

    state = ThompsonState(
        ds=ds_init,
        feature_params=params,
        true_w=w,
        max_fn_value=max_fn_value,
        argmax=argmax,
        noise_scale=noise_scale,
    )
    return state


@jax.jit
def update_state(key: PRNGKey, state: ThompsonState, x_besties: Array):
    """Update the current state by
    - adding 'x_besties' and corresponding objective function values to the state
    - adding 'x_besties' and 'y_besties' to data of the state
    - adding features for 'x_besties' to L
    - replacing current 'argmax' and 'max_fn_value' if a new maximum has been found
    """
    # x_besties (n_samples, D)
    # x_besties = jnp.reshape(
    #     x_besties, (x_besties.shape[0] * x_besties.shape[1], state.ds.D)
    # )
    # n_features = state.L.shape[-1]
    # L_besties = state.kernel.feature_fn(None, n_features, x_besties, recompute=False)
    # evaluate objective function at 'x_besties'
    y_besties = featurise(x_besties, state.feature_params) @ state.true_w
    y_besties = y_besties + jr.normal(key, shape=y_besties.shape) * state.noise_scale

    # add besties to state dataset
    x = jnp.concatenate([state.ds.x, x_besties], axis=0)
    y = jnp.concatenate([state.ds.y, y_besties], axis=0)
    N = state.ds.N + x_besties.shape[0]
    # construct updated state dataset
    ds = Dataset(x, y, N, state.ds.D)

    # add features of besties to state
    # L = jnp.concatenate([state.L, L_besties], axis=0)

    # find maximum of besties
    idx = jnp.argmax(y_besties)
    argmax, max_fn_value = x_besties[idx], y_besties[idx]
    # update maximum in state if appropriate

    max_fn_value, argmax = jax.lax.cond(
        max_fn_value <= state.max_fn_value,
        lambda: (state.max_fn_value, state.argmax),
        lambda: (max_fn_value, argmax),
    )
    # construct and return updated state
    updated_state = ThompsonState(
        ds=ds,
        # kernel=state.kernel,
        feature_params=state.feature_params,
        true_w=state.true_w,
        max_fn_value=max_fn_value,
        argmax=argmax,
        noise_scale=state.noise_scale,
    )
    return updated_state


def fake_dataset_like(dataset: Dataset):
    return Dataset(x=dataset.x[:1], y=dataset.y[:1], N=1, D=dataset.D)


def get_thompson_step_fn(
    thompson_config: ConfigDict,
    inference_config_representer: ConfigDict,
    inference_config_sample: ConfigDict,
    model: GPModel,
):
    if thompson_config.model_name == "RandomSearch":

        def _fn(key: PRNGKey, state: ThompsonState, i: Optional[int] = None):
            friends_key, noise_key = jr.split(key, 2)
            x_friends = find_friends(
                friends_key,
                state.ds.D,
                thompson_config.n_samples,
                method="uniform",
                minval=thompson_config.minval,
                maxval=thompson_config.maxval,
            )
            return update_state(noise_key, state, x_friends)

    else:
        gp_sample_argmax_f = jax.jit(
            partial(gp_sample_argmax, config=thompson_config, kernel=model.kernel)
        )
        def _fn(key: PRNGKey, state: ThompsonState, i: Optional[int] = None):
            representer_key, noise_key, friends_key, samples_key = jr.split(key, 4)

            # API compatibility dummy test set and features
            test_ds = fake_dataset_like(state.ds)

            # get posterior samples
            alpha_map = model.compute_representer_weights(
                key=representer_key,
                train_ds=state.ds,
                test_ds=test_ds,
                config=inference_config_representer,
                metrics_list=["loss", "err", "reg"],
                metrics_prefix=f"Thompson_{i}/alpha_MAP",
                exact_metrics=None,
                recompute=True,
            )

            if thompson_config.model_name == "SVGP":
                inducing_inputs = model.vi_params["variational_family"][
                    "inducing_inputs"
                ]
                L_train = featurise(
                    inducing_inputs,
                    state.feature_params,
                )
                L = L_train
                alpha_map = jnp.zeros(inducing_inputs.shape[0])
            else:
                L_train = featurise(state.ds.x, state.feature_params)
                inducing_inputs = None

                L_test = featurise(test_ds.x, state.feature_params)
                L = jnp.concatenate([L_train, L_test], axis=0)

            (
                _,
                zero_mean_alpha_samples,
                w_samples,
            ) = model.compute_posterior_samples(
                key=samples_key,
                n_samples=thompson_config.n_samples,
                train_ds=state.ds,
                test_ds=test_ds,
                config=inference_config_sample,
                use_rff=True,
                L=L,
                zero_mean=False,
                metrics_list=["loss", "err", "reg"],
                metrics_prefix=f"Thompson_{i}/alpha_samples",
            )

            # function optimisation starts here
            x_besties = gp_sample_argmax_f(
                key=friends_key,
                state=state,
                alpha_map=alpha_map,
                zero_mean_alpha_samples=zero_mean_alpha_samples,
                w_samples=w_samples,
                inducing_inputs=inducing_inputs,
                i=i,
            )

            return update_state(noise_key, state, x_besties)

    return _fn


def gp_sample_argmax(
    key: PRNGKey,
    state: ThompsonState,
    alpha_map: Array,
    zero_mean_alpha_samples: Array,
    w_samples: Array,
    inducing_inputs: Optional[Array],
    i: Optional[int],
    config: ConfigDict,
    kernel: Kernel,
) -> Array:
    acquisition_fn_sharex, acquisition_fn, acquisition_grad = get_acquisition_fn(
        state, kernel, alpha_map, zero_mean_alpha_samples, w_samples, inducing_inputs
    )

    # initial random search

    def scan_fn(key, ii):
        key, next_key = jax.random.split(key)
        x_friends = find_friends(
            key,
            config.D,
            config.n_friends,
            method=config.find_friends_method,
            minval=config.minval,
            maxval=config.maxval,
            state=state,
            lengthscale=kernel.kernel_config["length_scale"],
        )  #  ( config.n_friends, state.ds.D)

        y_friends = acquisition_fn_sharex(x_friends)  # (num_samples, config.n_friends)

        # select top random points
        x_homies = find_homies(x_friends, y_friends, config.n_homies)
        #  [n_samples, n_homies, D]
        return next_key, x_homies

    friend_idx = jnp.arange(config.friends_iterations)
    _, x_homies = jax.lax.scan(
        scan_fn, key, friend_idx
    )  # (friends_iterations, samples, n_friends, D)

    x_homies = (
        x_homies.transpose(0, 2, 1, 3)
        .reshape(-1, x_homies.shape[1], x_homies.shape[3])
        .transpose(1, 0, 2)  # (friends_iterations*n_friends, samples, D)
    )

    x_besties, _ = find_besties(
        x_homies,
        acquisition_fn,
        acquisition_grad,
        learning_rate=config.optim_lr,
        iterations=config.optim_iters,
        n_besties=config.n_besties,
        optim_trace=False,
        minval=config.minval,
        maxval=config.maxval,
        i=i,
    )

    return x_besties.reshape(-1, x_besties.shape[-1])


def find_friends(
    key: PRNGKey,
    ndims,
    n_friends: int,
    method: str = "uniform",
    minval: float = -1.0,
    maxval: float = 1.0,
    state: Optional[ThompsonState] = None,
    lengthscale: Optional[Array] = None,
) -> Array:
    """Given the current state, choose the next batch of exploration points."""

    if method == "uniform":
        x_friends = jr.uniform(
            key, shape=(n_friends, ndims), minval=minval, maxval=maxval
        )
    elif method == "nearby":
        num_explore = n_friends // 10
        num_exploit = 9 * n_friends // 10

        key_uniform, key_nearby, key_selector = jax.random.split(key, 3)
        x_friends_uniform = jr.uniform(
            key_uniform, shape=(num_explore, ndims), minval=minval, maxval=maxval
        )
        # TODO: consider making this uniform between - lengthscale[None, :] / 4 and lengthscale[None, :] / 4
        x_friends_localised_noise = jr.normal(key_nearby, shape=(num_exploit, ndims))
        x_friends_localised_noise = x_friends_localised_noise * lengthscale[None, :] / 2

        scores = state.ds.y + state.ds.y.min() + 1e-6
        scores = scores / scores.sum()
        indices = jax.random.choice(
            key_selector, len(scores), shape=(num_exploit,), replace=True, p=scores
        )
        x_friends_localised = state.ds.x[indices] + x_friends_localised_noise # num_exploit, ndims
        x_friends = jnp.concatenate([x_friends_uniform, x_friends_localised], axis=0)
        x_friends = jnp.clip(x_friends, a_min=minval, a_max=maxval)
    else:
        # TODO: implement other strategies
        raise NotImplementedError(
            f"Strategy '{method}' to find friends is not implemented."
        )
    return x_friends


def find_homies(x: Array, y: Array, n_homies: int):
    """For every sample, find the 'n_homies' x which produce the highest y."""

    idx = jnp.argsort(y, axis=-1)[:, -n_homies:]
    return x[idx]  # [n_samples, n_homies, D]


def find_besties(
    x_homies: Array,
    acquisition_fn: Callable,
    acquisition_grad: Callable,
    learning_rate: float = 1e-3,
    iterations: int = 100,
    n_besties: int = 1,
    optim_trace: bool = False,
    minval: float = -1.0,
    maxval: float = 1.0,
    i: Optional[int] = None,
):
    """For every sample, independently maximise the acqusition function value of 'n_homies' exploration points.
    Return the 'n_besties' x for each sample with highest acquisition function value

    Args:
        x_homies: (num_samples, num_homies, D)

    """

    optimiser = optax.adam(learning_rate=learning_rate)

    @jax.jit
    def update(x, opt_state):
        # pass -grad to maximise function
        updates, opt_state = optimiser.update(-acquisition_grad(x), opt_state, x)
        x = optax.apply_updates(x, updates)
        x = x.clip(min=minval, max=maxval)
        return x, opt_state

    opt_state = optimiser.init(x_homies)

    # TODO: will this work for (num_samples) y_homies?
    trace = []
    if optim_trace:
        y_homies = acquisition_fn(x_homies)
        trace.append((x_homies, y_homies))

    scan_idx = jnp.arange(iterations)

    def scan_fn(scan_state, ii):
        x_homies, opt_state, trace = scan_state
        x_homies, opt_state = update(x_homies, opt_state)
        if optim_trace:
            y_homies = acquisition_fn(x_homies)
            trace.append((x_homies, y_homies))
        return (x_homies, opt_state, trace), 0

    scan_state = jax.lax.scan(scan_fn, (x_homies, opt_state, trace), scan_idx)[0]
    x_homies, _, trace = scan_state

    if wandb.run is not None and optim_trace:
        for ii, step_data in enumerate(trace):
            _, y_homies = step_data
            wandb.log(
                {
                    **{f"Thompson_{i}/bestie_y": y_homies}
                    ** {f"Thompson_{i}/bestie_step": ii},
                }
            )

    @jax.vmap
    def top_args(x, y):
        """
        Args (after vmap):
            x: (n_samples, n_homies, D)
            y: (n_samples, n_homies)
        Returns (after vmap):
            (n_samples, n_besties, D)

        """
        return x[jnp.argsort(y)[-n_besties:]]

    if not optim_trace:
        y_homies = acquisition_fn(x_homies)

    return top_args(x_homies, y_homies), trace


def get_acquisition_fn(
    state: ThompsonState,
    kernel: Kernel,
    alpha_map: Array,  # (n_train,)
    alpha_samples: Array,  # (n_samples, n_train)
    w_samples: Array,  # (n_samples, n_features)
    inducing_inputs: Optional[Array] = None,
    **kernel_kwargs,
):
    """Construct single acquisition function which is vmapped over samples and inputs,
    returning element-wise function values and gradients.

    acquisition_fn_sharex:  (n_inputs, D) -> (n_samples, n_inputs)
    acquisition_fn:   (n_samples, n_inputs, D) -> (n_samples, n_inputs)
    acquisition_grad: (n_samples, n_inputs, D) -> (n_samples, n_inputs, D)
    """

    def _fn(x, alpha_sample, w_sample):
        # x: (D,)
        # alpha_sample: (n_train,)
        # w_sample: (n_features,)
        # return: ()
        L = featurise(x, state.feature_params)
        if inducing_inputs is None:
            K = kernel.kernel_fn(x, state.ds.x, **kernel_kwargs)
        else:
            K = kernel.kernel_fn(x, inducing_inputs, **kernel_kwargs)

        return (L @ w_sample + K @ (alpha_map - alpha_sample)).squeeze()

    @jax.jit
    def acquisition_fn_sharex(x):
        """
        in_shape: (n_inputs, D)
        out_shape: (n_samples, n_inputs)
        """
        return jax.vmap(jax.vmap(_fn, in_axes=(0, None, None)), in_axes=(None, 0, 0))(
            x, alpha_samples, w_samples
        )

    @jax.jit
    def acquisition_fn(x):
        """
        in_shape: (n_samples, n_inputs, D)
        out_shape: (n_samples, n_inputs)
        """
        return jax.vmap(jax.vmap(_fn, in_axes=(0, None, None)), in_axes=(0, 0, 0))(
            x, alpha_samples, w_samples
        )

    @jax.jit
    def acquisition_grad(x):
        """
        in_shape: (n_samples, n_inputs, D)
        out_shape: (n_samples, n_inputs, D)
        """
        return jnp.transpose(
            jax.vmap(
                jax.vmap(
                    jax.grad(_fn),
                    in_axes=(0, 0, 0),
                ),
                in_axes=(1, None, None),
            )(x, alpha_samples, w_samples),
            axes=(1, 0, 2),
        )

    return acquisition_fn_sharex, acquisition_fn, acquisition_grad


@partial(
    jax.jit,
    static_argnums=[
        3,
    ],
)
def grid_search(state: ThompsonState, minval=-1.0, maxval=1.0, grid_dim: int = 500):
    chex.assert_axis_dimension(state.ds.x, axis=1, expected=2)

    r = jnp.linspace(minval, maxval, num=grid_dim)
    xx, yy = jnp.meshgrid(r, r)
    grid = jnp.vstack([xx.ravel(), yy.ravel()])

    fn_value = featurise(grid.T, state.feature_params) @ state.true_w

    idx = jnp.argmax(fn_value)
    return fn_value[idx], grid.T[idx]  # max_fn_value, argmax
