import collections
from typing import Optional

import jax.numpy as jnp
import ml_collections
from chex import Array


def apply_z_score(data: Array, mu: Optional[Array]=None, sigma: Optional[Array]=None):
    if (mu is not None) and (sigma is not None):
        return (data - mu) / sigma
    else:
        mu = jnp.mean(data, axis=0)
        sigma = jnp.std(data, axis=0)
        return (data - mu) / sigma, mu, sigma


def revert_z_score(data: Array, mu: Array, sigma: Array):
    return sigma * data + mu


def RMSE(x: Array, x_hat: Array, mu: Optional[Array]=None, sigma: Optional[Array]=None):
    if mu is not None and sigma is not None:
        x = revert_z_score(x, mu, sigma)
        x_hat = revert_z_score(x_hat, mu, sigma)
    return jnp.sqrt(jnp.mean((x - x_hat) ** 2))


# Taken from https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
def flatten_nested_dict(nested_dict, parent_key="", sep="."):
    items = []
    for name, cfg in nested_dict.items():
        new_key = parent_key + sep + name if parent_key else name
        if isinstance(cfg, collections.MutableMapping):
            items.extend(flatten_nested_dict(cfg, new_key, sep=sep).items())
        else:
            items.append((new_key, cfg))

    return dict(items)


def update_config_dict(config_dict: ml_collections.ConfigDict, run, new_vals: dict):
    config_dict.unlock()
    config_dict.update_from_flattened_dict(run.config)
    config_dict.update_from_flattened_dict(new_vals)
    run.config.update(new_vals, allow_val_change=True)
    config_dict.lock()


if __name__ == '__main__':
    pass
