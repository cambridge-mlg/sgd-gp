import jax
import jax.numpy as jnp


@jax.jit
def RBF(x, y, s=1., l=1.):
    d2 = jnp.sum((x[:, None] - y[None, :]) ** 2, axis=-1)
    return (s ** 2) * jnp.exp(-.5 * d2 / (l ** 2))


if __name__ == '__main__':
    key = jax.random.PRNGKey(12345)
    N = 2
    M = 3
    D = 5

    key, key_x, key_y = jax.random.split(key, 3)
    X = jax.random.normal(key_x, (N, D))
    Y = jax.random.normal(key_y, (M, D))

    Z = RBF(X, Y)
