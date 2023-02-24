import jax
import jax.numpy as jnp


def error(params, targets, K):
    return .5 * jnp.sum((targets - K @ params) ** 2)


def regularizer(params, K):
    return .5 * params @ K @ params


def loss(params, targets, K):
    return error(params, targets, K) + regularizer(params, K)


def exact_solution(targets, K):
    return jnp.linalg.solve(K + jnp.identity(targets.shape[0]), targets)


def predict(params, x_pred, x_train, kernel_fn, **kernel_kwargs):
    return kernel_fn(x_pred, x_train, **kernel_kwargs) @ params


if __name__ == '__main__':
    key = jax.random.PRNGKey(12345)
    N = 10
    D = 5

    key, key_x, key_y = jax.random.split(key, 3)
    X = jax.random.normal(key_x, (N, D))
    Y = jax.random.normal(key_y, (N,))

    from kernels import RBF
    K = RBF(X, X)

    key, subkey = jax.random.split(key)
    alpha = jax.random.normal(subkey, (N,))
    print(error(alpha, Y, K))
    print(regularizer(alpha, K))

    alpha_exact = exact_solution(Y, K)
    print(alpha.shape, alpha_exact.shape)
    print(error(alpha_exact, Y, K))
    print(regularizer(alpha_exact, K))
