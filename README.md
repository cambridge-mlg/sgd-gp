# Stochastic Gradient Descent for Gaussian Processes

This repository contains code for our series of papers on using Stochastic Gradient Descent as a scalable algorithm for posterior inference in Gaussian Processes. Specifically, we provide code to run experiments from the following papers

1. Sampling from Gaussian Process Posteriors using Stochastic Gradient Descent (NeurIPS 2023 Oral) [[arXiv]](https://arxiv.org/abs/2306.11589) [[NeurIPS]](https://neurips.cc/virtual/2023/poster/71593) [[blog]](https://papers.avt.im/stochastic-gradient-descent-gp/)
2. Stochastic Gradient Descent for Gaussian Processes Done Right (ICLR 2024) [[arXiv]](https://arxiv.org/abs/2310.20581) [[OpenReview]](https://openreview.net/forum?id=fj2E5OcLFn)

For a quick start, you can refer to the extremely concise notebook [`sdd.ipynb`](https://github.com/cambridge-mlg/sgd-gp/blob/main/sdd.ipynb) to see an example of the SDD algorithm from our ICLR paper in action.

# UCI Regression Benchmarks

In order to replicate our UCI Regression experiments from both papers, you can run the following command to calculate the posterior mean.

```bash
python scalable_gps/scripts/obtain_mean.py --config configs/default.py:<uci_dataset> --config.method_name <method>
```

Here, you can specify the UCI dataset from the list in the `uci_datasets` repo, and the method from the list provided below. The script calculates the posterior mean, calculates metrics such as Test RMSE, and saves the posterior mean both as a Pickle file, and as a `wandb` artifact (if `config.wandb.log_artifact` is true). 

In order to obtain samples from the posterior, you can then run the following command

```bash
python scalable_gps/scripts/obtain_samples.py --config configs/default.py:<uci_dataset> --config.method_name <method>
```
The posterior mean is automatically saved as a `wandb` artifact, and loaded by the `obtain_samples.py` script. The script saves the posterior samples as a Pickle file, as well as a `wandb` artifact. Furthermore, the scripts should automatically load the tuned GP hyperparameters for all datasets from our `wandb` project. We replicate these hyperparameters in the [`configs/uci_regression_hparams.txt`](https://github.com/cambridge-mlg/sgd-gp/blob/main/scalable_gps/configs/uci_regression_hparams.txt) file.

We use UCI regression datasets from the excellent `uci_datasets` [repo](https://github.com/treforevans/uci_datasets), and you can use the same name as specified in the repo.

We provide parallelised Jax code for the following methods:

- `--config.method_name sgd --config.grad_variant vanilla` replicates the algorithm from our NeurIPS paper.
- `--config.method_name sgd --config.grad_variant batch_all` replicates the improved SDD algorithm from our ICLR paper.
- `--config.method_name cg` refers to conjugate gradients as a baseline. You can use the `--config.cg_config.preconditioner` flag to use a preconditioner.
- `--config.method_name vi` refers to an SVGP baseline.

# Molecular Benchmarks

In order to replicate our molecular benchmarks, you can specify the `--config.kernel_name` to use either the `TanimotoKernel` or `TanimotoL1Kernel`, and the dataset using `tanimoto_<target>`, where the targets include `esr2, f2, kit, parp1, pgr`. For example, you can run

```bash
python scalable_gps/scripts/obtain_mean.py --config configs/default.py:tanimoto_<target> --config.kernel_name TanimotoKernel --config.method_name <method>
```

# Thompson Sampling Experiments

To replicate our parallel Thompson sampling experiment on synthetic data sampled from ground truth GP, use the following command:

```bash
python scalable_gps/scripts/thompson_sampling.py
```

The default configuration file is `scalable_gps/configs/thompson_config.py`.

Use
- `--config.thompson.model_name CGGP` to run the conjugate gradients baseline
- `--config.thompson.model_name SVGP` to run the SVGP baseline
- `--config.thompson.model_name SGDGP` to run our SGD algorithm together with
  - `--config.train_config.grad_variant vanilla` for the algorithm from our NeurIPS paper or
  - `--config.train_config.grad_variant batch_all` for the improved SDD algorithm from our ICLR paper.

# Navigating the code

We implement baselines and our method in the `models` folder. Specifically, we inherit the [`GPModel`](https://github.com/cambridge-mlg/sgd-gp/blob/27af90a5bc4842c5b153dd40aded7cb4018490e0/scalable_gps/models/base_gp_model.py#L12) class that provides boilerplate code for defining a GP model. We then implement an exact GP posterior using a Cholesky decomposition, by implementing the [`compute_representer_weights()`](https://github.com/cambridge-mlg/sgd-gp/blob/27af90a5bc4842c5b153dd40aded7cb4018490e0/scalable_gps/models/exact_gp_model.py#L26) and [`compute_posterior_samples()`](https://github.com/cambridge-mlg/sgd-gp/blob/27af90a5bc4842c5b153dd40aded7cb4018490e0/scalable_gps/models/exact_gp_model.py#L90) functions. Finally, all our baselines overwrite these two functions.

The [`scripts`](https://github.com/cambridge-mlg/sgd-gp/tree/main/scalable_gps/scripts) folder contains the two scripts mentioned above, along with [`thompson_sampling.py`](https://github.com/cambridge-mlg/sgd-gp/blob/main/scalable_gps/scripts/thompson_sampling.py), to run our Thomspon sampling benchmark. 

The [`kernels.py`](https://github.com/cambridge-mlg/sgd-gp/blob/main/scalable_gps/kernels.py) file implements many common GP kernels, such as the RBF, Matern, and Tanimoto kernels. Since we use Random Features to draw samples from the prior, each kernel needs to implement a [`feature_fn`](https://github.com/cambridge-mlg/sgd-gp/blob/27af90a5bc4842c5b153dd40aded7cb4018490e0/scalable_gps/kernels.py#L118) function that returns the random features for the kernel.

The [`linear_model.py`](https://github.com/cambridge-mlg/sgd-gp/blob/main/scalable_gps/linear_model.py) file implements all the gradients for our SGD/SDD variants.

# References

Note: We would love to collaborate with anyone who is interested in using our code for their own research. Please feel free to reach out to us if you have any questions or need help with the code. 

We would also love to implement our method in your GP library of choice. Please feel free to reach out to us if you would like to collaborate on this!


If you use this repository to run SGD/SDD on your own experiments, please consider citing our papers:

```
@inproceedings{lin2023sampling,
    title = {Sampling from Gaussian Process Posteriors using Stochastic Gradient Descent},
    author = {Jihao Andreas Lin and Javier Antorán and Shreyas Padhy and David Janz and José Miguel Hernández-Lobato and Alexander Terenin},
    booktitle = {Advances in Neural Information Processing Systems},
    year = {2023}
}

@inproceedings{lin2024stochastic,
    title = {Stochastic Gradient Descent for Gaussian Processes Done Right},
    author = {Jihao Andreas Lin and Shreyas Padhy and Javier Antorán and Austin Tripp and Alexander Terenin and Csaba Szepesvári and José Miguel Hernández-Lobato and David Janz},
    booktitle = {International Conference on Learning Representations},
    year = {2024}
}
```
