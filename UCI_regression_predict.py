import os
import argparse
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import matplotlib.pyplot as plt
from utils import apply_z_score, RMSE
from kernels import RBF, RFF
from linear_model import exact_solution, predict, error_grad_sample, regularizer_grad_sample, loss_fn, draw_prior_function_sample, draw_prior_noise_sample
from uci_datasets import Dataset
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # random seed and device config
    parser.add_argument("--optimizer_seed", default=12345, type=int,
                        help="PyTorch initial random seed")
    parser.add_argument("--feature_seed", default=1234, type=int,
                        help="random seed for function sample")
    parser.add_argument("--sample_seed", default=123, type=int,
                        help="random seed for function sample")
    parser.add_argument("--device", default='cpu', type=str,
                        help="which compute device (usually 'cpu' or 'cuda')")
    # data config
    parser.add_argument("--datasets", default=['housing', 'concrete', 'wine', 'yacht'],
                        type=str, nargs='+',
                        help="which UCI regression datasets to use")
    # kernel hyperparameters
    parser.add_argument("--noise_scale", default=1., type=float,
                        help="observation noise scale")
    parser.add_argument("--signal_scale", default=1., type=float,
                        help="signal scale of the kernel")
    parser.add_argument("--length_scale", default=1., type=float,
                        help="length scale of the kernel")
    # optimization config
    parser.add_argument("--learning_rate", default=1e-2, type=float,
                        help="which learning rate to use")
    parser.add_argument("--momentum", default=.9, type=float,
                        help="amount of Nesterov momentum")
    parser.add_argument("--polyak", default=1e-2, type=float,
                        help="step size used for exponential Polyak averaging")
    parser.add_argument("--random_features", default=[1, 10, 100], type=int, nargs='+',
                        help="number of random features to approximate regularizer")
    parser.add_argument("--batch_sizes", default=[1, 2, 4, 8], type=int, nargs='+',
                        help="number of training example processed at once")
    parser.add_argument("--iterations", default=50000, type=int,
                        help="number of training iterations")
    # results path
    parser.add_argument("--results_path", default="plots/UCI/prior_sample_noise_in_error/lr=1e-2/{}.png", type=str,
                        help="filepath to results destination")
    args = parser.parse_args()
    results_dict = vars(args)
    
    kernel_fn = lambda x, y: RBF(x, y, s=args.signal_scale, l=args.length_scale)
    feature_fn = lambda key, n_features, x: RFF(key, n_features, x, s=args.signal_scale, l=args.length_scale)

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
        K = kernel_fn(x_train, x_train)
        
        # compute exact solution
        alpha_exact = exact_solution(y_train, K, noise_scale=args.noise_scale)

        # create figure
        rows = len(args.batch_sizes)
        cols = 4
        fig = plt.figure(figsize=[20, 5 * rows])
        fig.suptitle(f"{dataset_name}, N = {N}, D = {D}, lr = {args.learning_rate:.0e}")

        for row, B in enumerate(args.batch_sizes):
            assert B <= N
            ax = [fig.add_subplot(rows, cols, row * cols + idx + 1) for idx in range(cols)]

            for M in args.random_features:
                print(f"B = {B}, M = {M}")

                # create jax random keys for prior sample
                feature_key = jr.PRNGKey(args.feature_seed)
                prior_function_key, prior_noise_key = jr.split(jr.PRNGKey(args.sample_seed))

                # draw prior function sample evaluated at train and test data
                prior_function_sample_train = draw_prior_function_sample(feature_key, prior_function_key, M, x_train, feature_fn)
                prior_function_sample_test = draw_prior_function_sample(feature_key, prior_function_key, M, x_test, feature_fn)

                # draw prior noise sample
                prior_noise_sample = draw_prior_noise_sample(prior_noise_key, N, noise_scale=args.noise_scale)

                # compute exact solution
                alpha_sample_exact = exact_solution(prior_function_sample_train + prior_noise_sample, K, noise_scale=args.noise_scale)

                # compute exact prediction and RMSE of sample
                y_pred_sample_exact =  prior_function_sample_test + predict(alpha_exact - alpha_sample_exact, x_test, x_train, kernel_fn=kernel_fn)
                test_rmse_exact = RMSE(y_pred_sample_exact, y_test, mu=mu_y, sigma=sigma_y)

                # define auxiliary functions
                exact_loss_fn = jax.jit(lambda params: loss_fn(params, prior_function_sample_train + prior_noise_sample, K, noise_scale=args.noise_scale))

                @jax.jit
                def stochastic_gradient(params, key):
                    error_key, regularizer_key = jr.split(key)
                    error_grad = error_grad_sample(params, error_key, B, x_train, prior_function_sample_train + prior_noise_sample, kernel_fn)
                    regularizer_grad = regularizer_grad_sample(params, regularizer_key, M, x_train, feature_fn)
                    # error_grad = error_grad_sample(params, error_key, B, x_train, prior_function_sample_train, kernel_fn)
                    # regularizer_grad = regularizer_grad_sample(params - prior_noise_sample, regularizer_key, M, x_train, feature_fn)
                    return error_grad + (args.noise_scale ** 2) * regularizer_grad
                
                alpha = jnp.zeros((N,))
                alpha_polyak = jnp.zeros((N,))

                optimizer = optax.sgd(learning_rate=args.learning_rate, momentum=args.momentum, nesterov=True)
                opt_state = optimizer.init(alpha)

                @jax.jit
                def update(opt_state, params, params_polyak, key):
                    # use gradient of mean instead of sum over data for stability
                    grad = stochastic_gradient(params, key) / N
                    updates, opt_state = optimizer.update(grad, opt_state)
                    new_params = optax.apply_updates(params, updates)
                    new_params_polyak = optax.incremental_update(new_params, params_polyak, step_size=args.polyak)
                    return opt_state, new_params, new_params_polyak

                exact_loss_trace = []
                grad_var_trace = []
                alpha_rmse_trace = []
                test_rmse_trace = []

                key = jr.PRNGKey(args.optimizer_seed)
                iterator = tqdm(range(args.iterations))
                for i in iterator:
                    # perform update
                    key, subkey = jr.split(key)
                    opt_state, alpha, alpha_polyak = update(opt_state, alpha, alpha_polyak, subkey)

                    if i % 100 == 0:
                        # compute trace statistics
                        exact_loss = exact_loss_fn(alpha_polyak)
                        
                        y_pred_sample =  prior_function_sample_test + predict(alpha_exact - alpha_polyak, x_test, x_train, kernel_fn=kernel_fn)
                        alpha_rmse = RMSE(alpha_sample_exact, alpha_polyak)
                        test_rmse = RMSE(y_pred_sample, y_test, mu=mu_y, sigma=sigma_y)

                        grad_var_key = jr.split(jr.PRNGKey(12345), 100)
                        grad_samples = jax.vmap(stochastic_gradient, (None, 0))(alpha_polyak, grad_var_key)
                        grad_var = jnp.var(grad_samples, axis=0).mean()

                    exact_loss_trace.append(exact_loss.item())
                    grad_var_trace.append(grad_var.item())
                    alpha_rmse_trace.append(alpha_rmse.item())
                    test_rmse_trace.append(test_rmse.item())
                
                ax[0].plot(exact_loss_trace, label=f'M = {M}')
                ax[1].plot(grad_var_trace, label=f'M = {M}')
                ax[2].plot(alpha_rmse_trace, label=f'M = {M}')
                ax[3].plot(test_rmse_trace, label=f'M = {M}')

            ax[0].set_ylabel(f"Batch Size = {B}")
            ax[0].axhline(exact_loss_fn(alpha_sample_exact), color='k', linestyle='--', label='Exact')
            ax[2].axhline(0., color='k', linestyle='--', label='Exact')
            ax[3].axhline(test_rmse_exact, color='k', linestyle='--', label='Exact')
            ax[0].semilogy()
            ax[1].semilogy()
            ax[3].semilogy()

            if row == 0:
                ax[0].set_title("Exact Loss")
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
