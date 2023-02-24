import jax.numpy as jnp


def apply_z_score(data, mu=None, sigma=None):
    if (mu is not None) and (sigma is not None):
        return (data - mu) / sigma
    else:
        mu = jnp.mean(data, axis=0)
        sigma = jnp.std(data, axis=0)
        return (data - mu) / sigma, mu, sigma


def revert_z_score(data, mu, sigma):
    return sigma * data + mu


def RMSE(x, x_hat, mu=None, sigma=None):
    if mu is not None and sigma is not None:
        x = revert_z_score(x, mu, sigma)
        x_hat = revert_z_score(x_hat, mu, sigma)
    return jnp.sqrt(jnp.mean((x - x_hat) ** 2))


if __name__ == '__main__':
    pass
