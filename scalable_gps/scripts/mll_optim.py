"""Perform MLL Optimisation using Adam and exact gradients for ExactGPModel."""
import pickle

import jax
import jax.numpy as jnp
import jax.random as jr
import ml_collections.config_flags
import wandb
from absl import app, flags

from scalable_gps import kernels
from scalable_gps.baselines.exact_gp_model import ExactGPModel
from scalable_gps.data import get_dataset, subsample
from scalable_gps.utils import (
    HparamsTuple,
    flatten_nested_dict,
    setup_training,
    update_config_dict,
)

ml_collections.config_flags.DEFINE_config_file(
    "config",
    "configs/default.py",
    "Training configuration.",
    lock_config=True,
)

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
        setup_training(run)
        # If there are any config values dependent on sweep values, recompute them here.
        computed_configs = {}
        update_config_dict(config, run, computed_configs)

        print(config)
        train_ds, test_ds = get_dataset(config.dataset_name, **config.dataset_config)

        print(f"train_ds.x.shape: {train_ds.x.shape}")
        print(f"train_ds.y.shape: {train_ds.y.shape}")
        print(f"test_ds.x.shape: {test_ds.x.shape}")
        print(f"test_ds.y.shape: {test_ds.y.shape}")

        train_subsample_key, test_subsample_key = jr.split(
            jr.PRNGKey(config.mll_config.subsample_seed)
        )
        train_ds = subsample(
            train_subsample_key, train_ds, config.mll_config.n_subsample
        )
        test_ds = subsample(test_subsample_key, test_ds, config.mll_config.n_subsample)

        print(f"subsampled train_ds.x.shape: {train_ds.x.shape}")
        print(f"subsampled train_ds.y.shape: {train_ds.y.shape}")
        print(f"subsampled test_ds.x.shape: {test_ds.x.shape}")
        print(f"subsampled test_ds.y.shape: {test_ds.y.shape}")

        hparams = HparamsTuple(
            length_scale=jnp.array(config.mll_config.init_length_scale),
            signal_scale=config.mll_config.init_signal_scale,
            noise_scale=config.mll_config.init_noise_scale,
        )

        kernel_init_fn = getattr(kernels, config.kernel_name)

        kernel = kernel_init_fn(
            {"signal_scale": hparams.signal_scale, "length_scale": hparams.length_scale}
        )
        exact_model = ExactGPModel(hparams.noise_scale, kernel)

        hparams = exact_model.compute_mll_optim(
            hparams,
            train_ds,
            config.mll_config,
            test_ds,
            None,
            transform=jax.nn.softplus,
            perform_eval=True,
        )

        # Use wandb artifacts to save model hparams for a given dataset split and subsample_idx.
        hparams_artifact = wandb.Artifact(
            f"hparams_{config.dataset_name}_{config.dataset_config.split}_{config.mll_config.subsample_seed}",
            type="hparams",
            description=f"Model hparams for {config.dataset_name} dataset with subsample seed {config.mll_config.subsample_seed} on split {config.dataset_config.split}.",
            metadata={
                **{
                    "dataset_name": config.dataset_name,
                    "split": config.dataset_config.split,
                },
                **config.mll_config,
            },
        )

        with hparams_artifact.new_file("hparams.pkl", "wb") as f:
            pickle.dump(hparams, f)

        wandb.log_artifact(hparams_artifact)


if __name__ == "__main__":
    import os
    import sys

    if sys.argv:
        # pass wandb API as argv[1] and set environment variable
        # 'python mll_optim.py MY_API_KEY'
        os.environ["WANDB_API_KEY"] = sys.argv[1]

    # Adds jax flags to the program.
    jax.config.config_with_absl()

    # Snippet to pass configs into the main function.
    def _main(argv):
        del argv
        config = FLAGS.config
        main(config)

    app.run(_main)
