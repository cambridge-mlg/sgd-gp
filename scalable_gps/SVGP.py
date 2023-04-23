import jax
import jax.numpy as jnp
import gpjax as gpx
import jaxutils
from jax import random
from jaxutils import Dataset as jaxDataset
import jaxkern as jk
from jax import jit
from functools import partial
import chex
from scalable_gps.data import Dataset


def freeze_non_variational_params(parameter_state):
    """Modifies parameter_state in-place and also returns parameter_state"""
    non_trainables = jax.tree_util.tree_map(lambda x: False, parameter_state.trainables)
    non_trainables["variational_family"] = jax.tree_util.tree_map(
        lambda x: True, non_trainables["variational_family"]
    )
    parameter_state.trainables = non_trainables
    return parameter_state


from scalable_gps.kmeans import kmeans


def regression_SVGP(
    train_dataset: Dataset,
    num_inducing: int,
    kernel_name: str,
    kernel_config: chex.ArrayTree,
    ARD: bool,
    noise_scale: float,
    key: chex.PRNGKey,
    inducing_init: str = "kmeans",
):
    """Returns a function that computes the negative ELBO of a GP regression model with
    SVGP inference together with initialised variational parameters and a function to make
    predictions."""
    D = jaxDataset(X=train_dataset.x, y=train_dataset.y[:, None])

    jaxutils.config.reset_global_config()
    likelihood = gpx.Gaussian(num_datapoints=D.n)
    kernel_class = get_jk_kernel_from_name(kernel_name)

    if ARD:
        prior = gpx.Prior(
            kernel=kernel_class(active_dims=list(range(train_dataset.x.shape[1])))
        )
    else:
        prior = gpx.Prior(kernel=kernel_class())

    p = prior * likelihood

    if inducing_init == "kmeans":
        z, _ = kmeans(key, train_dataset.x, k=num_inducing, thresh=1e-3)
    elif inducing_init == "equidistant":
        chex.assert_axis_dimension(D.X, axis=1, expected=1)
        z = jnp.linspace(D.X.min(), D.X.max(), num_inducing).reshape(-1, 1)
    elif inducing_init == "uniform":
        z = random.uniform(
            key, shape=(num_inducing, train_dataset.x.shape[1]), minval=0.0, maxval=1.0
        )
        max_val = train_dataset.x.max(axis=0, keepdims=True)
        min_val = train_dataset.x.min(axis=0, keepdims=True)
        z = z * (max_val - min_val) + min_val

    q = gpx.VariationalGaussian(prior=prior, inducing_inputs=z)
    svgp = gpx.StochasticVI(posterior=p, variational_family=q)

    negative_elbo = jit(svgp.elbo(D, negative=True))

    parameter_state = gpx.initialise(
        svgp,
        key,
        likelihood={"obs_noise": jnp.array([noise_scale**2])},
        kernel={
            "variance": kernel_config["signal_scale"] ** 2,
            "lengthscale": kernel_config["length_scale"],
        },
    )  # GPS works with variances instead of scales for all but lengthscale
    parameter_state = freeze_non_variational_params(parameter_state)

    def get_predictive(learned_params, test_x, eps=1e-4):
        """Returns multivariate Gaussian posterior distributions over function
        evaluations (eps noise std) and over observations."""
        function_dist = q(learned_params)(test_x)
        predictive_dist = likelihood(learned_params, function_dist)
        return (
            likelihood({"likelihood": {"obs_noise": jnp.array([eps])}}, function_dist),
            predictive_dist,
        )

    return negative_elbo, parameter_state, D, get_predictive


def sample_from_qu(key: chex.PRNGKey, learned_params: chex.ArrayTree, num_samples: int):
    """Sample from variational distribution over inducing observations."""
    mu = learned_params["variational_family"]["moments"]["variational_mean"]
    L = learned_params["variational_family"]["moments"]["variational_root_covariance"]
    eps = random.normal(key, shape=(mu.shape[0], num_samples))
    return mu + L @ eps


# NOTE: sampling optimisable functions for Thompson sampling can be done with the pathwwise
# method by sampling u at inducing locations z=learned_params["variational_family"]['inducing_inputs']
# with the above method and then sampling f | u ~ f(.) + K(.,z)K(zz)^-1(u-f(z))


def get_jk_kernel_from_name(kernel_name):
    """Returns a jaxkern kernel from a string name."""
    if kernel_name == "RBFKernel":
        return jk.RBF
    elif kernel_name == "Matern12Kernel":
        return jk.Matern12
    elif kernel_name == "Matern32Kernel":
        return jk.Matern32
    elif kernel_name == "Matern52Kernel":
        return jk.Matern52
    else:
        raise ValueError(f"Unsupported kernel name {kernel_name}")
