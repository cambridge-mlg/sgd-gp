# Stochastic Gradient Descent for Gaussian Processes

This repository contains code for our series of papers on using Stochastic Gradient Descent as a scalable algorithm for posterior inference in Gaussian Processes. Specifically, we provide code to run experiments from the following papers

1. Sampling from Gaussian Process Posteriors using Stochastic Gradient Descent (NeurIPS 2023 Oral) [[arXiv]](https://arxiv.org/abs/2306.11589) [[NeurIPS]](https://neurips.cc/virtual/2023/poster/71593) [[blog]](https://papers.avt.im/stochastic-gradient-descent-gp/)
2. Stochastic Gradient Descent for Gaussian Processes Done Right (ICLR 2024) [[arXiv]](https://arxiv.org/abs/2310.20581) [[OpenReview]](https://openreview.net/forum?id=fj2E5OcLFn)

For a quick start, you can refer to the extremely concise notebook `sdd.ipynb` to see an example of the SDD algorithm from our ICLR paper in action.

# UCI Regression Benchmarks

In order to replicate our UCI Regression experiments from both papers, you can run the following command to calculate the posterior mean.

```bash
python scripts/obtain_mean.py --config configs/default.py:<uci_dataset> --config.method_name <method>
```

In order to obtain samples from the posterior, you can then run the following command

```bash
python scripts/obtain_samples.py --config configs/default.py:<uci_dataset> --config.method_name <method>
```
We use `wandb` for logging, and the posterior mean is automatically saved as a `wandb` artifact, and loaded by the `obtain_samples.py` script. Furthermore, the scripts should automatically load the tuned GP hyperparameters for all datasets from our `wandb` project. We replicate these hyperparameters in the `configs/uci_regression_hparams.txt` file.

We use UCI regression datasets from the excellent `uci_datasets` [repo](https://github.com/treforevans/uci_datasets), and you can use the same name as specified in the repo.

We provide parallelised Jax code for the following methods:

1. `--config.method_name sgd --config.grad_variant vanilla` replicates the algorithm from our NeurIPS paper.
2. `--config.method_name sgd --config.grad_variant batch_kvp` replicates the improved SDD algorithm from our ICLR paper.
3. `--config.method_name cg` refers to conjugate gradients as a baseline. You can use the `--config.cg_config.preconditioner` flag to use a preconditioner.
4. `--config.method_name vi` refers to an SVGP baseline.

# Molecular Benchmarks

In order to replicate our molecular benchmarks, you can specify the `--config.kernel_name` to use either the `TanimotoKernel` or `TanimotoL1Kernel`, and the dataset using `tanimoto_<target>`, where the targets include `esr2, f2, kit, parp1, pgr`.

# Navigating the code

We implement baselines and our method in the `models` folder. Specifically, we inherit the `GPModel` class that provides boilerplate code for defining a GP model. We then implement an exact GP posterior using a Cholesky decomposition, by implementing the `compute_representer_weights()` and `compute_posterior_samples()` functions. Finally, all our baselines overwrite these two functions.

The `scripts` folder contains the two scripts mentioned above, along with `thompson_sampling.py`, to run our Thomspon sampling benchmark. 

The `kernels.py` implements many common GP kernels, such as the RBF, Matern, and Tanimoto kernels. Since we use Random Features to draw samples from the prior, each kernel needs to implement a `feature_fn` function that returns the random features for the kernel.

The `linear_model.py` file implements all the gradients for our SGD/SDD variants.

# References

If you use this repository to run SGD/SDD on your own experiments, please consider citing our papers:

```
@article{lin2024sampling,
  title={Sampling from gaussian process posteriors using stochastic gradient descent},
  author={Lin, Jihao Andreas and Antor{\'a}n, Javier and Padhy, Shreyas and Janz, David and Hern{\'a}ndez-Lobato, Jos{\'e} Miguel and Terenin, Alexander},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}

@article{lin2023stochastic,
  title={Stochastic Gradient Descent for Gaussian Processes Done Right},
  author={Lin, Jihao Andreas and Padhy, Shreyas and Antor{\'a}n, Javier and Tripp, Austin and Terenin, Alexander and Szepesv{\'a}ri, Csaba and Hern{\'a}ndez-Lobato, Jos{\'e} Miguel and Janz, David},
  journal={arXiv preprint arXiv:2310.20581},
  year={2023}
}
```

Note: We would love to collaborate with anyone who is interested in using our code for their own research. Please feel free to reach out to us if you have any questions or need help with the code. 

We would also love to implement our method in your GP library of choice. Please feel free to reach out to us if you would like to collaborate on this!
```
