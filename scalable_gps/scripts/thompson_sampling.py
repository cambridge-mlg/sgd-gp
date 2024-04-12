"""Run Thompson Sampling experiments using different GP models."""
import jax
import jax.numpy as jnp
import jax.random as jr
import ml_collections.config_flags
import time
import wandb
from absl import app, flags
from tqdm import tqdm

from scalable_gps import kernels
from scalable_gps import thompson_utils
from scalable_gps.models.exact_gp_model import ExactGPModel
from scalable_gps.models.vi_gp_model import SVGPThompsonInterface
from scalable_gps.models.cg_gp_model import CGGPModel
from scalable_gps.models.sgd_gp_model import SGDGPModel
from scalable_gps.utils import flatten_nested_dict, setup_training, update_config_dict

ml_collections.config_flags.DEFINE_config_file(
    "config",
    "configs/thompson_config.py",
    "Thompson sampling configuration.",
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
        computed_configs["train_config.polyak"] = (
            100 / run.config["train_config.iterations"]
        )
        computed_configs["sampling_config.polyak"] = (
            100 / run.config["sampling_config.iterations"]
        )
        update_config_dict(config, run, computed_configs)
        print(config)

        print("Initialising kernel and model...")
        kernel_init_fn = getattr(kernels, config.thompson.kernel_name)
        kernel = kernel_init_fn(
            {
                "signal_scale": config.thompson.signal_scale,
                "length_scale": jnp.array([config.thompson.length_scale]),
            }
        )
        kernel.kernel_config["use_ard"] = False

        key = jr.PRNGKey(config.thompson.seed)
        optim_key, init_key = jr.split(key)
        print("Constructing initial state of Thompson sampling")
        state = thompson_utils.init(
            key=init_key,
            D=config.thompson.D,
            kernel=kernel,
            noise_scale=config.thompson.noise_scale,
            n_features=config.thompson.n_features,
            n_init=config.thompson.n_init,
            minval=config.thompson.minval,
            maxval=config.thompson.maxval,
            init_method=config.thompson.init_method,
        )
        print(f"Initial max_fn_value = {state.max_fn_value}")
        # print(f"Initial argmax = {state.argmax}")

        if config.thompson.grid_search:
            grid_max_fn_value, grid_argmax = thompson_utils.grid_search(
                state, grid_dim=config.thompson.grid_search_dim
            )
            print(f"Grid max_fn_value = {grid_max_fn_value}")
            print(f"Grid argmax = {grid_argmax}")
        else:
            grid_max_fn_value, grid_argmax = None, None

        if wandb.run is not None:
            wandb.log(
                {
                    "thompson_step": 0,
                    "max_fn_value": state.max_fn_value,
                    # "argmax": state.argmax,
                    "thompson_time": 0.0,
                    "n_observed": state.ds.x.shape[0],
                    "grid_max_fn_value": grid_max_fn_value,
                    "grid_argmax": grid_argmax,
                }
            )

        train_config, sampling_config, model = None, None, None
        if config.thompson.model_name == "ExactGP":
            model = ExactGPModel(config.thompson.noise_scale, kernel)
        elif config.thompson.model_name == "CGGP":
            model = CGGPModel(config.thompson.noise_scale, kernel)
            train_config, sampling_config = config.cg_config, config.cg_config
        elif config.thompson.model_name == "SGDGP":
            model = SGDGPModel(config.thompson.noise_scale, kernel)
            train_config, sampling_config = config.train_config, config.sampling_config
        elif config.thompson.model_name == "SVGP":
            model = SVGPThompsonInterface(config.thompson.noise_scale, kernel, config)
            train_config, sampling_config = config.vi_config, config.vi_config

        step_fn = thompson_utils.get_thompson_step_fn(
            config.thompson, train_config, sampling_config, model
        )

        print("running model ", config.thompson.model_name)

        key = optim_key
        start_time = time.time()
        for i in tqdm(range(config.thompson.iterations)):
            key, step_key = jr.split(key)
            state = step_fn(step_key, state, i)

            print(f"thompson_step {i + 1} ")
            print(f"n_observed {state.ds.x.shape[0]} ")
            print(f"max_fn_value {state.max_fn_value} ")
            # print(f"max_location {state.argmax} ")
            print(f"thompson_time {time.time() - start_time} ")

            if wandb.run is not None:
                wandb.log(
                    {
                        "n_observed": state.ds.x.shape[0],
                        "thompson_step": i + 1,
                        "max_fn_value": state.max_fn_value,
                        "thompson_time": time.time() - start_time,
                    }
                )

        print(f"Final max_fn_value = {state.max_fn_value}")
        # print(f"Final argmax = {state.argmax}")


if __name__ == "__main__":
    import os
    import sys

    if sys.argv:
        # pass wandb API as argv[1] and set environment variable
        # 'python thompson_sampling.py MY_API_KEY'
        os.environ["WANDB_API_KEY"] = sys.argv[1]

    # Adds jax flags to the program.
    jax.config.config_with_absl()

    # Snippet to pass configs into the main function.
    def _main(argv):
        del argv
        config = FLAGS.config
        main(config)

    app.run(_main)
