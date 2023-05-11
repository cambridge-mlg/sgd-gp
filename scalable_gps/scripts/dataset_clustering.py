import pickle

import jax
import jax.numpy as jnp
import jax.random as jr
import ml_collections.config_flags
import wandb
from absl import app, flags
import time

from scalable_gps import kernels

from scalable_gps.data import get_dataset, subsample
from scalable_gps.utils import (
    HparamsTuple,
    flatten_nested_dict,
    setup_training,
    update_config_dict,
    get_tuned_hparams,
)
from scalable_gps.knn import annoy_cluster_dataset


ml_collections.config_flags.DEFINE_config_file(
    "config",
    "configs/clustering_config.py",
    "Clustering configuration.",
    lock_config=True,
)

import numpy as np

FLAGS = flags.FLAGS


def main(config):
    wandb_kwargs = {
        "project": config.wandb.project,
        "entity": config.wandb.entity,
        "name": config.wandb.name if config.wandb.name else None,
        "config": flatten_nested_dict(config.to_dict()),
        "mode": "online" if config.wandb.log else "disabled",
        "settings": wandb.Settings(code_dir=config.wandb.code_dir),
    }
    with wandb.init(**wandb_kwargs) as run:
        # setup_training(run)
        # If there are any config values dependent on sweep values, recompute them here.
        computed_configs = {}
        update_config_dict(config, run, computed_configs)

        print(config)
        train_ds, test_ds = get_dataset(config.dataset_name, split=config.dataset_split)
        try:
            hparams = get_tuned_hparams(config.dataset_name, split=config.dataset_split)
        except wandb.CommError:
            print("Could not fetch hparams from wandb. Using default values.")

        lengthscales = hparams.length_scale
        max_dist = (
            np.percentile(lengthscales**2, config.lengthscale_percentile)
            / config.lengthscale_ratio
        )

        print("lengthscales", lengthscales)
        print("max_dist", max_dist)

        print(f"train_ds.x.shape: {train_ds.x.shape}")
        print(f"train_ds.y.shape: {train_ds.y.shape}")
        print(f"test_ds.x.shape: {test_ds.x.shape}")
        print(f"test_ds.y.shape: {test_ds.y.shape}")

        tic = time.time()
        he_train_z, keep_indices = annoy_cluster_dataset(
            train_ds,
            n_trees=config.n_trees,
            num_neighbours=config.n_neighbours,
            max_dist=max_dist,
            savefile=None,
            recompute=False,
        )
        toc = time.time()
        print("time", toc - tic)

        hparams_artifact = wandb.Artifact(
            f"clustered_indices_{config.dataset_name}_{config.dataset_split}_lengthscale_ratio_{config.lengthscale_ratio}",
            type="hparams",
            description=f"Model hparams for {config.dataset_name} dataset with subsample"
            f"on split {config.dataset_split} with lengthscale ratio {config.lengthscale_ratio}.",
            metadata={
                **{
                    "dataset_name": config.dataset_name,
                    "split": config.dataset_split,
                    "lengthscale_ratio": config.lengthscale_ratio,
                },
            },
        )

        with hparams_artifact.new_file("indices.pkl", "wb") as f:
            pickle.dump(keep_indices, f)

        wandb.log_artifact(hparams_artifact)


if __name__ == "__main__":
    import os
    import sys

    # if sys.argv:
    # pass wandb API as argv[1] and set environment variable
    # 'python mll_optim.py MY_API_KEY'
    # os.environ["WANDB_API_KEY"] = sys.argv[1]

    # Adds jax flags to the program.
    # jax.config.config_with_absl()

    # Snippet to pass configs into the main function.
    def _main(argv):
        del argv
        config = FLAGS.config
        main(config)

    app.run(_main)
