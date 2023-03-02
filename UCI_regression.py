import os
import argparse
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from utils import apply_z_score, RMSE
from kernels import RBF
from linear_model import exact_solution, predict, error, regularizer
from uci_datasets import Dataset
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # random seed and device config
    parser.add_argument("--seed", default=12345, type=int,
                        help="PyTorch initial random seed")
    parser.add_argument("--device", default='cpu', type=str,
                        help="which compute device (usually 'cpu' or 'cuda')")
    # data config
    parser.add_argument("--datasets", default=['housing', 'concrete', 'wine', 'yacht'],
                        type=str, nargs='+',
                        help="which UCI regression datasets to use")
    # optimization config
    parser.add_argument("--learning_rates", default=[1e-2, 1e-3, 1e-4], type=float, nargs='+',
                        help="which learning rates to use")
    parser.add_argument("--momentum", default=.9, type=float,
                        help="amount of Nesterov momentum")
    parser.add_argument("--polyak", default=1e-2, type=float,
                        help="step size used for exponential Polyak averaging")
    parser.add_argument("--batch_sizes", default=[1, 2, 4, 8], type=int, nargs='+',
                        help="number of training example processed at once")
    parser.add_argument("--iterations", default=50000, type=int,
                        help="number of training iterations")
    # results path
    parser.add_argument("--results_path", default="plots/UCI/{}.png", type=str,
                        help="filepath to results destination")
    args = parser.parse_args()
    results_dict = vars(args)
    

    for dataset_name in args.datasets:
        # load data
        dataset = Dataset(dataset_name)
        x_train, y_train, x_test, y_test = dataset.get_split(0)
        N, D = x_train.shape
        T = x_test.shape[0]

        # convert to jax and apply z-score normalization
        x_train, mu_x, sigma_x = apply_z_score(jnp.array(x_train))
        y_train, mu_y, sigma_y = apply_z_score(jnp.array(y_train.squeeze()))
        x_test = apply_z_score(jnp.array(x_test), mu=mu_x, sigma=sigma_x)
        y_test = apply_z_score(jnp.array(y_test.squeeze()), mu=mu_y, sigma=sigma_y)
        # compute kernel matrix
        K = RBF(x_train, x_train)

        # compute exact solution
        alpha_exact = exact_solution(y_train, K)
        y_pred_exact = predict(alpha_exact, x_test, x_train, RBF)
        test_rmse_exact = RMSE(y_pred_exact, y_test, mu=mu_y, sigma=sigma_y)

        # define auxiliary functions
        @jax.jit
        def loss_fn(params):
            return error(params, y_train, K) + regularizer(params, K)

        @jax.jit
        def error_grad_sample(params, i):
            return -K[i, :] * (y_train[i] - K[i, :] @ params) * N

        @jax.jit
        def stochastic_gradient(params, i):
            return error_grad_sample(params, i) + K @ params
        
        @jax.jit
        def batch_gradient(params, batch_idx):
            return jax.vmap(stochastic_gradient, (None, 0))(params, batch_idx)

        @jax.jit
        def gradient_variance(params):
            grad_samples = batch_gradient(params, jnp.arange(N))
            return jnp.sum(jnp.var(grad_samples, axis=0)) # computes trace of covariance matrix

        # create figure
        rows = len(args.batch_sizes)
        cols = 4
        fig = plt.figure(figsize=[20, 5 * rows])
        fig.suptitle(f"{dataset_name}, N = {N}, D = {D}")

        for row, batch_size in enumerate(args.batch_sizes):
            assert batch_size <= N
            ax = [fig.add_subplot(rows, cols, row * cols + idx + 1) for idx in range(cols)]
                
            # perform optimization
            for lr in args.learning_rates:
                print(f"batch_size = {batch_size}, learning_rate = {lr:.0e}")
                alpha = jnp.zeros((N,))
                alpha_polyak = jnp.zeros((N,))

                optimizer = optax.sgd(learning_rate=lr, momentum=args.momentum, nesterov=True)
                opt_state = optimizer.init(alpha)

                @jax.jit
                def update(opt_state, params, params_polyak, key):
                    i = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=N)
                    grad = jnp.mean(batch_gradient(params, i), axis=0) / N # normalize gradient by dataset size to make learning rate agnostic
                    updates, opt_state = optimizer.update(grad, opt_state)
                    new_params = optax.apply_updates(params, updates)
                    new_params_polyak = optax.incremental_update(new_params, params_polyak, step_size=args.polyak)
                    return opt_state, new_params, new_params_polyak

                loss_trace = []
                grad_var_trace = []
                alpha_rmse_trace = []
                test_rmse_trace = []

                key = jax.random.PRNGKey(args.seed)
                iterator = tqdm(range(args.iterations))
                for _ in iterator:
                    # perform update
                    key, subkey = jax.random.split(key)
                    opt_state, alpha, alpha_polyak = update(opt_state, alpha, alpha_polyak, subkey)
                    # compute trace statistics
                    loss = loss_fn(alpha_polyak)
                    grad_var = gradient_variance(alpha_polyak) / batch_size
                    y_pred = predict(alpha_polyak, x_test, x_train, RBF)

                    loss_trace.append(loss.item())
                    grad_var_trace.append(grad_var.item())
                    alpha_rmse_trace.append(RMSE(alpha_exact, alpha_polyak).item())
                    test_rmse_trace.append(RMSE(y_pred, y_test, mu=mu_y, sigma=sigma_y))
                
                ax[0].plot(loss_trace, label=f'lr={lr:.0e}')
                ax[1].plot(grad_var_trace, label=f'lr={lr:.0e}')
                ax[2].plot(alpha_rmse_trace, label=f'lr={lr:.0e}')
                ax[3].plot(test_rmse_trace, label=f'lr={lr:.0e}')

            ax[0].set_ylabel(f"Batch Size = {batch_size}")
            ax[0].axhline(loss_fn(alpha_exact), color='k', linestyle='--', label='Exact')
            ax[2].axhline(0., color='k', linestyle='--', label='Exact')
            ax[3].axhline(test_rmse_exact, color='k', linestyle='--', label='Exact')
            ax[0].semilogy()
            ax[1].semilogy()
            ax[3].semilogy()

            if row == 0:
                ax[0].set_title("Loss")
                ax[1].set_title("Gradient Variance")
                ax[2].set_title(r"$\alpha$ RMSE")
                ax[3].set_title("Test RMSE")
            
            for idx in range(cols):
                ax[idx].grid(alpha=.5)
                ax[idx].legend()
                if row == len(args.batch_sizes) - 1:
                    ax[idx].set_xlabel("Iterations")

        fig.tight_layout()

        path = args.results_path.format(dataset_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, bbox_inches='tight')
