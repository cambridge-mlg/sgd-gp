
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
        "config": flatten_nested_dict(config.to_dict()),
        "mode": "online" if config.wandb.log else "disabled",
        "settings": wandb.Settings(code_dir=config.wandb.code_dir),
    }

    with wandb.init(**wandb_kwargs) as run:
        setup_training(run)
        # If there are any config values dependent on sweep values, recompute them here.
        computed_configs = {}
        update_config_dict(config, run, computed_configs)
        
        print("Initialising kernel and model...")
        kernel_init_fn = getattr(kernels, config.kernel_name)
        kernel = kernel_init_fn({'signal_scale': config.signal_scale, 'length_scale': jnp.array(config.length_scale)})
        model = ExactGPModel(config.noise_scale, kernel)

        key = jr.PRNGKey(config.seed)
        key, init_key = jr.split(key)
        print("Constructing initial state of Thompson sampling")
        state = thompson_utils.init(init_key, config.D, kernel, config.n_features, config.n_init)
        print(f"Initial max_fn_value = {state.max_fn_value}")
        if wandb.run is not None:
            wandb.log({'train_step': 0, 'max_fn_value': state.max_fn_value, 'wall_clock_time': 0.0})

        start_time = time.time()
        for i in tqdm(range(config.iterations)):
            alpha_map = model.compute_representer_weights(state.ds, recompute=True)

            key, friends_key, samples_key = jr.split(key, 3)
            ds_friends = thompson_utils.find_friends(friends_key, state, config.n_friends, method=config.find_friends_method)
            L = thompson_utils.add_features(state.L, ds_friends.x, state.feature_fn)

            ds_friends.y, alpha_samples, w_samples = model.compute_posterior_samples(samples_key, config.n_samples, state.ds, ds_friends, use_rff=True, L=L, zero_mean=False)
            
            acquisition_fn = thompson_utils.get_acquisition_fn(state, alpha_map, alpha_samples, w_samples, kernel.kernel_fn)
            
            x_homies = thompson_utils.find_homies(ds_friends, config.n_homies)
            x_besties, _ = thompson_utils.find_besties(x_homies, acquisition_fn, learning_rate=config.optim_lr, iterations=config.optim_iters, n_besties=config.n_besties)
            
            state = thompson_utils.update_state(state, x_besties)
            
            if wandb.run is not None:
                wandb.log({'train_step': i + 1, 'max_fn_value': state.max_fn_value, 'wall_clock_time': time.time() - start_time})

        print(f"Final max_fn_value = {state.max_fn_value}")


if __name__ == "__main__":
    # Adds jax flags to the program.
    jax.config.config_with_absl()

    # Snippet to pass configs into the main function.
    def _main(argv):
        del argv
        config = FLAGS.config
        main(config)

    app.run(_main)