from dataclasses import dataclass
from typing import Any, Optional, Tuple

import jax.numpy as jnp
import jax.random as jr
from chex import Array
from uci_datasets import Dataset as uci_dataset
from uci_datasets import all_datasets

from utils import apply_z_score

PRNGKey = Any
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



def get_toy_sin_dataset(
    key: PRNGKey, 
    n: int, 
    noise: float, 
    n_test: int, 
    n_periods: int = 25, 
    normalise: bool = False,
    **kwargs: KwArgs
    ) -> Tuple[Dataset, Dataset]:
    k1, k2, key = jr.split(key, 3)

    x = jnp.linspace( -n / n_periods, n / n_periods, num = n).reshape(-1,1)

    def f(x):
        return jnp.sin(2 * x) + jnp.cos(5 * x)

    signal = f(x)
    y = signal + jr.normal(k2, shape=signal.shape) * noise

    x_test = jnp.linspace(-3.1, 3.1, 500).reshape(-1, 1)
    y_test = f(x_test)

    train_ds = Dataset(x, y, n, 1)
    test_ds = Dataset(x_test, y_test, n_test, 1)

    if normalise:
        train_ds, test_ds = _normalise_dataset(train_ds, test_ds)

    return train_ds, test_ds


def get_uci_dataset(
    dataset_name: str, normalise: bool = False, **kwargs: KwArgs) -> Tuple[Dataset, Dataset]:

    dataset = uci_dataset(dataset_name)
    x_train, y_train, x_test, y_test = dataset.get_split(0)
    N, D = x_train.shape
    N_test, _ = x_test.shape

    train_ds = Dataset(jnp.array(x_train), jnp.array(y_train).squeeze(), N, D)
    test_ds = Dataset(jnp.array(x_test), jnp.array(y_test).squeeze(), N_test, D)

    if normalise:
        train_ds, test_ds = _normalise_dataset(train_ds, test_ds)
    
    return train_ds, test_ds


def _normalise_dataset(
    train_ds: Dataset, test_ds: Dataset) -> Tuple[Dataset, Dataset]:

    train_ds.x, train_ds.mu_x, train_ds.sigma_x = apply_z_score(train_ds.x)
    train_ds.y, train_ds.mu_y, train_ds.sigma_y = apply_z_score(train_ds.y.squeeze())
    test_ds.x = apply_z_score(test_ds.x, mu=train_ds.mu_x, sigma=train_ds.sigma_x)
    test_ds.y = apply_z_score(test_ds.y.squeeze(), mu=train_ds.mu_y, sigma=train_ds.sigma_y)

    return train_ds, test_ds


def get_dataset(dataset_name, **kwargs):
    if dataset_name == 'toy_sin':
        return get_toy_sin_dataset(**kwargs)
    elif dataset_name in all_datasets.keys():
        return get_uci_dataset(dataset_name, **kwargs)
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')