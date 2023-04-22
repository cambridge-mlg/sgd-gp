from dataclasses import dataclass
from typing import Any, Optional, Tuple

import chex
import jax.numpy as jnp
import jax.random as jr
from chex import Array
from uci_datasets import Dataset as uci_dataset
from uci_datasets import all_datasets
from scalable_gps.utils import apply_z_score

KwArgs = Any


@dataclass
class Dataset:
    x: Array
    y: Array
    N: int
    D: int

    mu_x: Optional[Array] = None
    sigma_x: Optional[Array] = None
    mu_y: Optional[Array] = None
    sigma_y: Optional[Array] = None


def subsample(key: chex.PRNGKey, ds: Dataset, n_subsample: int = 10000):
    # return full dataset if it is smaller than subsample size
    if ds.N <= n_subsample:
        return ds

    # choose random data point
    i = jr.randint(key, (), minval=0, maxval=ds.N)

    # compute Euclidean distances to each data point
    # (could be replaced by any other distance measure if desired)
    dist = jnp.sum((ds.x[i, None] - ds.x[None, :]) ** 2, axis=-1).squeeze()
    idx = jnp.argsort(dist)[:n_subsample]

    # create subsampled dataset
    ds_subsample = Dataset(
        x=ds.x[idx],
        y=ds.y[idx],
        N=n_subsample,
        D=ds.D,
        mu_x=ds.mu_x,
        sigma_x=ds.sigma_x,
        mu_y=ds.mu_y,
        sigma_y=ds.sigma_y,
    )

    return ds_subsample


def get_concentrating_toy_sin_dataset(
    seed: int,
    n: int,
    noise_scale: float,
    n_test: int,
    x_std: float = 1.0,
    **kwargs: KwArgs,
) -> Tuple[Dataset, Dataset]:
    key = jr.PRNGKey(seed)  # Required because configdict can't pass jr.PRNGKey as seed
    k1, k2, key = jr.split(key, 3)

    x = jr.normal(k1, shape=(n, 1)) * x_std

    def f(x):
        return jnp.squeeze(jnp.sin(2 * x) + jnp.cos(5 * x))

    signal = f(x)
    y = signal + jr.normal(k2, shape=signal.shape) * noise_scale

    x_test = jnp.linspace(-10.1, 10.1, n_test).reshape(-1, 1)
    y_test = f(x_test)

    train_ds = Dataset(x, y, n, 1)
    test_ds = Dataset(x_test, y_test, n_test, 1)

    return train_ds, test_ds


def get_expanding_toy_sin_dataset(
    seed: int,
    n: int,
    noise_scale: float,
    n_test: int,
    n_periods: int = 25,
    **kwargs: KwArgs,
) -> Tuple[Dataset, Dataset]:
    x = jnp.linspace(-n / n_periods, n / n_periods, num=n).reshape(-1, 1)

    def f(x):
        return jnp.squeeze(jnp.sin(2 * x) + jnp.cos(5 * x))

    signal = f(x)
    key = jr.PRNGKey(seed)
    y = signal + jr.normal(key, shape=signal.shape) * noise_scale

    x_test = jnp.linspace(-10.1, 10.1, n_test).reshape(-1, 1)
    y_test = f(x_test)

    train_ds = Dataset(x, y, n, 1)
    test_ds = Dataset(x_test, y_test, n_test, 1)

    return train_ds, test_ds


def get_uci_dataset(
    dataset_name: str, split: int = 0, **kwargs: KwArgs
) -> Tuple[Dataset, Dataset]:
    dataset = uci_dataset(dataset_name)
    x_train, y_train, x_test, y_test = dataset.get_split(split)
    N, D = x_train.shape
    N_test, _ = x_test.shape

    train_ds = Dataset(jnp.array(x_train), jnp.array(y_train), N, D)
    test_ds = Dataset(jnp.array(x_test), jnp.array(y_test), N_test, D)

    return train_ds, test_ds


def _normalise_dataset(train_ds: Dataset, test_ds: Dataset) -> Tuple[Dataset, Dataset]:
    train_ds.x, train_ds.mu_x, train_ds.sigma_x = apply_z_score(train_ds.x)
    train_ds.y, train_ds.mu_y, train_ds.sigma_y = apply_z_score(train_ds.y)
    test_ds.x = apply_z_score(test_ds.x, mu=train_ds.mu_x, sigma=train_ds.sigma_x)
    test_ds.y = apply_z_score(test_ds.y, mu=train_ds.mu_y, sigma=train_ds.sigma_y)

    return train_ds, test_ds


def get_dataset(dataset_name, **kwargs):
    if dataset_name == "toy_sin":
        train_ds, test_ds = get_expanding_toy_sin_dataset(**kwargs)
    elif dataset_name in all_datasets.keys():
        train_ds, test_ds = get_uci_dataset(dataset_name, **kwargs)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    train_ds.y = train_ds.y.squeeze()
    test_ds.y = test_ds.y.squeeze()
    chex.assert_rank([train_ds.y, test_ds.y], [1, 1])

    if kwargs.get("normalise", False):
        train_ds, test_ds = _normalise_dataset(train_ds, test_ds)

    return train_ds, test_ds
