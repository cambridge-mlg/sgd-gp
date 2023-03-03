import jax
import jax.numpy as jnp
import jax.random as jr


@jax.jit
def RBF(x, y, s=1., l=1.):
    d2 = jnp.sum((x[:, None] - y[None, :]) ** 2, axis=-1)
    return (s ** 2) * jnp.exp(-.5 * d2 / (l ** 2))


def RFF(key, n_features, x, s=1., l=1.):
    # compute single random Fourier feature for RBF kernel
    D = x.shape[-1]
    M = n_features
    omega_key, phi_key = jr.split(key)

    omega = jr.normal(omega_key, shape=(D, M))
    phi = jr.uniform(phi_key, shape=(1, M), minval=-jnp.pi, maxval=jnp.pi)
    return s * jnp.sqrt(2. / M) * jnp.cos(x @ (omega / l) + phi)


if __name__ == '__main__':
    key = jax.random.PRNGKey(12345)
    N = 10
    M = 3
    D = 5

    key, key_x, key_y = jax.random.split(key, 3)
    X = jax.random.normal(key_x, (N, D))

    Y = RFF(key_y, M, X)
    print(Y.shape)
    print(Y)
