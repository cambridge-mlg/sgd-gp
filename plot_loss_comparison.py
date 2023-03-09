import os
import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from uci_datasets import Dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data config
    parser.add_argument("--datasets", default=['housing', 'concrete', 'wine', 'yacht'],
                        type=str, nargs='+',
                        help="which UCI regression datasets to use")
    # results path
    parser.add_argument("--results_path", default="plots/UCI/loss_comparison/lr=1e-2_polyak=1e-3/{}.npy", type=str,
                        help="filepath to results destination")
    args = parser.parse_args()
    results_dict = vars(args)
    

    for dataset_name in args.datasets:
        dataset = Dataset(dataset_name)
        x_train, y_train, x_test, y_test = dataset.get_split(0)
        N, D = x_train.shape

        traces = jnp.load(args.results_path.format(dataset_name), allow_pickle=True).item()
        batch_sizes = traces.keys()
        # create figure
        rows = len(batch_sizes)
        cols = 4
        fig = plt.figure(figsize=[20, 5 * rows])
        fig.suptitle(f"{dataset_name}, N = {N}, D = {D}, lr = 1e-2")

        for row, B in enumerate(batch_sizes):
            assert B <= N
            ax = [fig.add_subplot(rows, cols, row * cols + idx + 1) for idx in range(cols)]
            for j in range(3):
                
                ax[0].plot(traces[B][j]['exact_loss'], label=f'Loss {j + 1}')
                ax[1].plot(traces[B][j]['grad_var'], label=f'Loss = {j + 1}')
                ax[2].plot(traces[B][j]['alpha_rmse'], label=f'Loss = {j + 1}')
                ax[3].plot(traces[B][j]['test_rmse'], label=f'Loss = {j + 1}')

            ax[0].set_ylabel(f"Batch Size = {B}")
            ax[0].axhline(0., color='k', linestyle='--', label='Exact')
            ax[2].axhline(0., color='k', linestyle='--', label='Exact')
            ax[3].axhline(0., color='k', linestyle='--', label='Exact')

            if row == 0:
                ax[0].set_title("Exact Loss")
                ax[1].set_title("Gradient Variance")
                ax[2].set_title(r"$\alpha$ RMSE")
                ax[3].set_title("Test RMSE")
            
            for idx in range(cols):
                ax[idx].grid(alpha=.5)
                ax[idx].legend()
                if row == len(batch_sizes) - 1:
                    ax[idx].set_xlabel("Iterations")

        fig.tight_layout()

        path = args.results_path.format(dataset_name).replace('npy', 'png')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, bbox_inches='tight')
