import jax
import jax.numpy as jnp
import jax.random as jr


def error(params, targets, K):
    return .5 * jnp.sum((targets - K @ params) ** 2)

# TODO: pmap over B.
# TODO: pass idx directly, can be different sizes depending on pmap and vmap.
def error_grad_sample(params, key, B, x, target, kernel_fn):

    N = x.shape[0]
    idx = jr.randint(key, shape=(B,), minval=0, maxval=N)
    K = kernel_fn(x[idx], x)
    
    print(f'shapes : {K.shape}, {target[idx].shape}')
    return -K.T @ (target[idx] - K @ params) * (N / B)


def regularizer(params, target, K, noise_scale):
    params = (noise_scale ** 2) * params
    return .5 * (params - target) @ K @ (params - target)


# TODO: pmap over M.
def regularizer_grad_sample(params, key, M, x, target, feature_fn, noise_scale):
    R = feature_fn(key, M, x)
    params = (noise_scale ** 2) * params
    return R @ (R.T @ (params - target))


def loss_fn(params, target_tuple, K, noise_scale=1.):
    error_target, regularizer_target = target_tuple
    return error(params, error_target, K) + regularizer(params, regularizer_target, K, noise_scale)


def exact_solution(targets, K, noise_scale=1.):
    return jnp.linalg.solve(K + (noise_scale ** 2) * jnp.identity(targets.shape[0]), targets)


def predict(params, x_pred, x_train, kernel_fn, **kernel_kwargs):
    return kernel_fn(x_pred, x_train, **kernel_kwargs) @ params


def draw_prior_function_sample(feature_key, prior_function_key, M, x, feature_fn, **feature_kwargs):
    R = feature_fn(feature_key, M, x, **feature_kwargs)
    w = jr.normal(prior_function_key, (M,))
    return R @ w


def draw_prior_noise_sample(prior_noise_key, N, noise_scale=1.):
    return noise_scale * jr.normal(prior_noise_key, (N,))


if __name__ == '__main__':
    key = jax.random.PRNGKey(12345)
    import matplotlib.pyplot as plt
    import numpy as np

    from kernels import RBF, RFF

    D = 5


    L = 30
    heatmap = np.zeros((L, L))
    r = np.logspace(1, 3, num=L).astype(int)

    for i, N in enumerate(r):
        for j, M in enumerate(r):
            print(N, M)
            key, subkey = jax.random.split(key)
            X = jax.random.normal(subkey, (N, D))

            s = 1.
            l = 1.
            K = RBF(X, X, s=s, l=l)

            seeds = 30
            for _ in range(seeds):
                key, subkey = jax.random.split(key)
                R = RFF(subkey, M, X, s=s, l=l)

                heatmap[i, j] += (((R @ R.T) - K) ** 2).mean() / seeds
    
    fig = plt.figure(figsize=[7, 7])
    ax = fig.add_subplot(111)
    ax.imshow(heatmap, cmap='inferno')
    ax.set_xlabel("M")
    ax.set_xticklabels(r)
    ax.set_ylabel("N")
    ax.set_yticklabels(r)
    plt.show()
            
    
