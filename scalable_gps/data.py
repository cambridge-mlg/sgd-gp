from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Optional, Tuple

import chex
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd
from chex import Array
from uci_datasets import Dataset as uci_dataset
from uci_datasets import all_datasets

import scalable_gps.fingerprint_utils as ff
from scalable_gps.utils import apply_z_score

KwArgs = Any


class ThompsonDataset(NamedTuple):
    x: Array
    y: Array
    N: int
    D: int

    z: Optional[Array] = None


@dataclass
class Dataset:
    x: Array
    y: Array
    N: int
    D: int

    z: Optional[Array] = None

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


def get_split_toy_sin_dataset(
    seed: int,
    n: int,
    noise_scale: float,
    n_test: int,
    x_std: float = 1.0,
    separation: float = 0.0,
    **kwargs: KwArgs,
) -> Tuple[Dataset, Dataset]:
    key = jr.PRNGKey(seed)  # Required because configdict can't pass jr.PRNGKey as seed
    k1, k2, key = jr.split(key, 3)

    x = jr.normal(k1, shape=(n, 1)) * x_std
    x = x.at[n // 2 :].set(x[n // 2 :] + separation / 2)
    x = x.at[: n // 2].set(x[: n // 2] - separation / 2)

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


def _normalise_protein_dataset(
    train_ds: Dataset, test_ds: Dataset, mean
) -> Tuple[Dataset, Dataset]:
    # Don't normalise hashed features for protein datasets.
    train_ds.mu_y, test_ds.mu_y = mean, mean
    train_ds.sigma_y, test_ds.sigma_y = 1.0, 1.0
    train_ds.y = apply_z_score(train_ds.y, mu=train_ds.mu_y, sigma=train_ds.sigma_y)
    test_ds.y = apply_z_score(test_ds.y, mu=train_ds.mu_y, sigma=train_ds.sigma_y)

    return train_ds, test_ds


def get_dataset(dataset_name, **kwargs):
    if dataset_name == "toy_sin":
        train_ds, test_ds = get_expanding_toy_sin_dataset(**kwargs)
    elif dataset_name in all_datasets.keys():
        train_ds, test_ds = get_uci_dataset(dataset_name, **kwargs)
    elif "tanimoto" in dataset_name:
        target = dataset_name.split("_")[1]
        train_ds, test_ds = get_protein_dataset(target, **kwargs)

    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    train_ds.y = train_ds.y.squeeze()
    test_ds.y = test_ds.y.squeeze()
    chex.assert_rank([train_ds.y, test_ds.y], [1, 1])

    if kwargs.get("normalise", False):
        if "tanimoto" in dataset_name:
            mean_y = kwargs.get("data_target_mean", None)
            print(f"mean y is {mean_y}")
            if mean_y is not None:
                train_ds, test_ds = _normalise_protein_dataset(
                    train_ds, test_ds, mean_y
                )
        else:
            train_ds, test_ds = _normalise_dataset(train_ds, test_ds)

    return train_ds, test_ds


def load_dockstring_dataset(
    dataset_dir: str, limit_num_train: Optional[int] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the dockstring dataset from the specified directory.

    Args:
        dataset_dir: Path to the directory containing the dockstring dataset.

    Returns:
        train_df, test_df: DataFrames containing the training and test data.
    """
    print("Starting to load datasets...")

    # Ensure file paths are present
    print(dataset_dir)
    dataset_path = Path(dataset_dir) / "dockstring-dataset.tsv"
    assert dataset_path.exists()

    dataset_split_path = Path(dataset_dir) / "cluster_split.tsv"
    assert dataset_split_path.exists()

    # Copied from data loading notebook
    df = pd.read_csv(dataset_path, sep="\t").set_index("inchikey")
    splits = (
        pd.read_csv(dataset_split_path, sep="\t")
        .set_index("inchikey")  # use same index as dataset
        .loc[df.index]  # re-order to match the dataset
    )

    df_train = df[splits["split"] == "train"]
    df_test = df[splits["split"] == "test"]

    # Optionally limit train data size by subsampling without replacement
    if limit_num_train is not None:
        assert limit_num_train <= len(df_train)
        df_train = df_train.sample(n=limit_num_train, replace=False)

    print("Finished loading datasets.")
    return df_train, df_test


def get_protein_dataset(
    target: str,
    dataset_dir: str = "",
    binarize=False,
    input_dim: int = 1,
    n_train: Optional[int] = None,
    **kwargs,
):
    df_train, df_test = load_dockstring_dataset(
        str(Path(dataset_dir)), limit_num_train=n_train
    )
    df_train = df_train[["smiles", target]].dropna()  # no nans
    df_test = df_test[["smiles", target]].dropna()  # no nans

    # Extract train/test SMILES
    smiles_train = df_train.smiles.to_list()
    smiles_test = df_test.smiles.to_list()
    y_train = df_train[target].to_numpy()
    y_test = df_test[target].to_numpy()

    # Clip to max of 5.0
    y_train = np.minimum(y_train, 5.0)
    y_test = np.minimum(y_test, 5.0)

    fp_kwargs = dict(use_counts=True, radius=1, nbits=input_dim, binarize=binarize)
    fp_train = ff.smiles_to_fingerprint_arr(smiles_train, **fp_kwargs).astype(
        np.float64
    )
    fp_test = ff.smiles_to_fingerprint_arr(smiles_test, **fp_kwargs).astype(np.float64)

    fp_train = jnp.array(fp_train)
    fp_test = jnp.array(fp_test)
    y_train = jnp.array(y_train)
    y_test = jnp.array(y_test)

    train_ds = Dataset(fp_train, y_train, len(y_train), input_dim)
    test_ds = Dataset(fp_test, y_test, len(y_test), input_dim)
    return train_ds, test_ds
