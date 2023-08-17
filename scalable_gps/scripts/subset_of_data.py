# pol dataset, N=15000, d=26
# train: 13500, test: 1500

# elevators dataset, N=16599, d=18
# train: 14940, test: 1659

# bike dataset, N=17379, d=17
# train: 15642, test: 1737

# protein dataset, N=45730, d=9
# train: 41157, test: 4573

# keggdirected dataset, N=48827, d=20
# train: 43945, test: 4882

# 3droad dataset, N=434874, d=3
# train: 391387, test: 43487

# song dataset, N=515345, d=90
# train: 463811, test: 51534

# buzz dataset, N=583250, d=77
# train: 524925, test: 58325

# houseelectric dataset, N=2049280, d=11
# train: 1844352, test: 204928

from scalable_gps.data import get_dataset
from scalable_gps.data import Dataset
from scalable_gps.utils import get_tuned_hparams
from scalable_gps.eval_utils import RMSE, mean_LLH
from scalable_gps.models.exact_gp_model import ExactGPModel
from scalable_gps import kernels
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import wandb

seed = 12345
key = jr.PRNGKey(seed)
# datasets = ['protein', 'keggdirected']
# datasets = ['3droad', 'song', 'buzz', 'houseelectric']
datasets = ['buzz', 'houseelectric']
kernel_name = "Matern32Kernel"

n_subsets = 6
n_seeds = 5
n_samples = 64
file = "./SoD_big_datasets.txt"

for dataset in datasets:

    if dataset == '3droad':
        splits = [0, 1, 2, 4]
    elif dataset == 'houseelectric':
        splits = [0, 1, 2]
    else:
        splits = [0, 1, 2, 3, 4]
    
    normalised_test_rmse = np.zeros((len(splits), n_subsets, n_seeds))
    normalised_test_nll = np.zeros((len(splits), n_subsets, n_seeds))
    normalised_test_nll_samples = np.zeros((len(splits), n_subsets, n_seeds))

    for i, split in enumerate(splits):
        # load full dataset
        data_train, data_test = get_dataset(dataset, split=split, normalise=True)

        # subset_sizes = [int(p * data_train.N) for p in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75]]
        subset_sizes = [1000, 2500, 5000, 10000, 25000, 50000]

        # get hparams
        try:
            hparams = get_tuned_hparams(dataset, split)
        except wandb.CommError:
            print(f"Could not fetch hparams from wandb for dataset '{dataset}', split '{split}'.")

        # init kernel
        kernel_init_fn = getattr(kernels, kernel_name)
        kernel = kernel_init_fn({'signal_scale': hparams.signal_scale, 'length_scale': hparams.length_scale})

        for j, subset_size in enumerate(subset_sizes):
            # select subset of data
            assert subset_size <= data_train.N
            for k in range(n_seeds):
                key, subset_key, sampling_key = jr.split(key, 3)
                idx = jr.permutation(subset_key, data_train.N)[:subset_size]

                # create dataset using subset
                data_train_subset = Dataset(
                    x=data_train.x[idx],
                    y=data_train.y[idx],
                    N=subset_size,
                    D=data_train.D,
                    mu_x=data_train.mu_x,
                    mu_y=data_train.mu_y,
                    sigma_x=data_train.sigma_x,
                    sigma_y=data_train.sigma_y)

                exact_model = ExactGPModel(hparams.noise_scale, kernel)
                exact_model.compute_representer_weights(data_train_subset)
                y_pred = exact_model.predictive_mean(data_train_subset, data_test)
                normalised_test_rmse[i, j] = RMSE(data_test.y, y_pred)

                batch_size = 10000
                start = 0
                y_pred_variance = jnp.zeros((0,))
                batch_idx = list(range(0, data_test.N, batch_size)) + [data_test.N]
                for b in range(len(batch_idx) - 1):
                    start = batch_idx[b]
                    end = batch_idx[b + 1]

                    data_test_batch = Dataset(
                        x=data_test.x[start:end],
                        y=data_test.y[start:end],
                        N=end-start,
                        D=data_test.D,
                        mu_x=data_test.mu_x,
                        mu_y=data_test.mu_y,
                        sigma_x=data_test.sigma_x,
                        sigma_y=data_test.sigma_y
                    )

                    y_pred_variance_batch = exact_model.predictive_variance(data_train_subset, data_test_batch, add_likelihood_noise=True)
                    y_pred_variance = jnp.append(y_pred_variance, y_pred_variance_batch)

                # y_pred_variance = exact_model.predictive_variance(data_train_subset, data_test, add_likelihood_noise=True)

                normalised_test_nll[i, j, k] = -mean_LLH(data_test.y, y_pred, y_pred_variance)

                # posterior_samples, _, _ = exact_model.compute_posterior_samples(
                #     sampling_key,
                #     n_samples=n_samples,
                #     train_ds=data_train_subset,
                #     test_ds=data_test,
                #     use_rff=False,
                #     zero_mean=True,
                # )
            
                # y_pred_variance_samples = exact_model.predictive_variance_samples(posterior_samples, add_likelihood_noise=True)
                normalised_test_nll_samples[i, j, k] = 0#-mean_LLH(data_test.y, y_pred, y_pred_variance_samples)
        
        
    rmse_mean = np.mean(normalised_test_rmse, axis=(0, 2))
    rmse_stderr = np.std(normalised_test_rmse, axis=(0, 2)) / np.sqrt(len(splits))

    nll_mean = np.mean(normalised_test_nll, axis=(0, 2))
    nll_stderr = np.std(normalised_test_nll, axis=(0, 2)) / np.sqrt(len(splits))

    nll_samples_mean = np.mean(normalised_test_nll_samples, axis=(0, 2))
    nll_samples_stderr = np.std(normalised_test_nll_samples, axis=(0, 2)) / np.sqrt(len(splits))

    print(f"dataset: {dataset}")
    for j, subset_size in enumerate(subset_sizes):
        print(f"  subset_size: {subset_size}, rmse: {rmse_mean[j]:.2f} +/- {rmse_stderr[j]:.2f}, nll: {nll_mean[j]:.2f} +/- {nll_stderr[j]:.2f}, nll_samples: {nll_samples_mean[j]:.2f} +/- {nll_samples_stderr[j]:.2f}")
    print()

    with open(file, 'a') as f:
        f.write(f"dataset: {dataset}\n")
        for j, subset_size in enumerate(subset_sizes):
            f.write(f"  subset_size: {subset_size}, rmse: {rmse_mean[j]:.2f} +/- {rmse_stderr[j]:.2f}, nll: {nll_mean[j]:.2f} +/- {nll_stderr[j]:.2f}, nll_samples: {nll_samples_mean[j]:.2f} +/- {nll_samples_stderr[j]:.2f}\n")
        f.write("\n")
