import os
from collections.abc import MutableMapping
from typing import List, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
import ml_collections
import wandb
from chex import Array


class TargetTuple(NamedTuple):
    error_target: Array
    regularizer_target: Array


class ExactPredictionsTuple(NamedTuple):
    alpha: Array
    y_pred_loc: Array


class ExactSamplesTuple(NamedTuple):
    alpha_sample: Array
    posterior_sample: Array
    alpha_map: Array
    f0_sample_test: Array


class HparamsTuple(NamedTuple):
    noise_scale: float
    signal_scale: float
    length_scale: Union[float, Array]


def get_gpu_or_cpu_device():
    if jax.default_backend() == "gpu":
        return jax.devices("gpu")[0]
    else:
        return jax.devices("cpu")[0]


def apply_z_score(
    data: Array, mu: Optional[Array] = None, sigma: Optional[Array] = None
):
    if (mu is not None) and (sigma is not None):
        return (data - mu) / sigma
    else:
        mu = jnp.mean(data, axis=0, keepdims=True)
        sigma = jnp.std(data, axis=0, keepdims=True)
        return (data - mu) / sigma, mu, sigma


def revert_z_score(data: Array, mu: Array, sigma: Array):
    return sigma * data + mu


# Taken from https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
def flatten_nested_dict(nested_dict, parent_key="", sep="."):
    items = []
    for name, cfg in nested_dict.items():
        new_key = parent_key + sep + name if parent_key else name
        if isinstance(cfg, MutableMapping):
            items.extend(flatten_nested_dict(cfg, new_key, sep=sep).items())
        else:
            items.append((new_key, cfg))

    return dict(items)


def update_config_dict(config_dict: ml_collections.ConfigDict, run, new_vals: dict):
    config_dict.unlock()
    config_dict.update_from_flattened_dict(run.config)
    config_dict.update_from_flattened_dict(new_vals)
    run.config.update(new_vals, allow_val_change=True)
    config_dict.lock()


def setup_training(wandb_run):
    """Helper function that sets up training configs and logs to wandb."""
    if not wandb_run.config.use_tpu:
        # # TF can hog GPU memory, so we hide the GPU device from it.
        # tf.config.experimental.set_visible_devices([], "GPU")

        # Without this, JAX is automatically using 90% GPU for pre-allocation.
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"
        # os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"
        # Disable logging of compiles.
        jax.config.update("jax_log_compiles", False)

        # Log various JAX configs to wandb, and locally.
        wandb_run.summary.update(
            {
                "jax_process_index": jax.process_index(),
                "jax.process_count": jax.process_count(),
            }
        )
    else:
        # config.FLAGS.jax_xla_backend = "tpu_driver"
        # config.FLAGS.jax_backend_target = os.environ['TPU_NAME']
        # DEVICE_COUNT = len(jax.local_devices())
        print(jax.default_backend())
        print(jax.device_count(), jax.local_device_count())
        print("8 cores of TPU ( Local devices in Jax ):")
        print("\n".join(map(str, jax.local_devices())))


def get_tuned_hparams(d_name: str, split: int):
    n_seeds = 10
    import pickle

    api = wandb.Api()

    noise_scales = []
    signal_scales = []
    length_scales = []

    for i in range(n_seeds):
        hparams_artifact_name = f"hparams_{d_name}_{split}_{i}"

        artifact = api.artifact(
            f"shreyaspadhy/scalable-gps/{hparams_artifact_name}:latest"
        )
        import sys

        from scalable_gps import utils

        sys.modules["utils"] = utils

        data = pickle.load(open(artifact.file(), "rb"))
        noise_scales.append(data.noise_scale)
        signal_scales.append(data.signal_scale)
        length_scales.append(data.length_scale)

    mean_hparams = HparamsTuple(
        noise_scale=float(jnp.mean(jnp.array(noise_scales))),
        signal_scale=float(jnp.mean(jnp.array(signal_scales))),
        length_scale=jnp.mean(jnp.array(length_scales), axis=0),
    )

    return mean_hparams


def get_clustered_indices(d_name: str, split: int, lengthscale_ratio: float):
    import pickle

    api = wandb.Api()

    name = f"clustered_indices_{d_name}_{split}_lengthscale_ratio_{lengthscale_ratio}"
    artifact = api.artifact(f"shreyaspadhy/scalable-gps/{name}:latest")
    import sys

    from scalable_gps import utils

    sys.modules["utils"] = utils

    data = pickle.load(open(artifact.file(), "rb"))
    return jnp.array(data)


def get_map_solution(
    d_name: str, method_name: str, split: int, override_noise_scale: float
):
    api = wandb.Api()
    import pickle
    import sys

    from scalable_gps import utils

    sys.modules["utils"] = utils

    artifact_name = f"alpha_{d_name}_{method_name}_{split}"
    if override_noise_scale > 0.0:
        artifact_name += f"_noise_{override_noise_scale}"

    artifact = api.artifact(f"shreyaspadhy/scalable-gps/{artifact_name}:latest")
    data = pickle.load(open(artifact.file(), "rb"))
    print(f'loaded {artifact_name}')
    return data

def get_latest_saved_artifact(
    artifact_name: str, all_save_steps: List[int]):
    
    api = wandb.Api()
    import pickle
    import sys

    from scalable_gps import utils

    sys.modules["utils"] = utils
    
    data = None
    artifact_loaded = False

    for i in all_save_steps:
        saved_artifact_name = f"{artifact_name}_{i}"
        
        try:
            artifact = api.artifact(f"shreyaspadhy/scalable-gps/{saved_artifact_name}:latest")
            data = pickle.load(open(artifact.file(), "rb"))
            artifact_loaded = True
            print(f'Loaded {saved_artifact_name}')
        except wandb.CommError:
            print(f'Failed to load {saved_artifact_name}')
            if artifact_loaded:
                break

    if data is None:
        print("No artifacts found.")
    
    return data

def save_latest_artifact(
    artifact_data, artifact_name):
    
    import pickle
    import sys

    from scalable_gps import utils

    sys.modules["utils"] = utils

    save_artifact = wandb.Artifact(
        artifact_name,
        type="saved_ckpt",
        description="saved checkpoint",)
                
    with save_artifact.new_file("saved_ckpt.pkl", "wb") as f:
        pickle.dump(artifact_data, f)
    
    wandb.log_artifact(save_artifact)

def process_pmapped_and_vmapped_metrics(pmapped_and_vmapped_metrics):
    mean_metrics, std_metrics = {}, {}
    for k, v in pmapped_and_vmapped_metrics.items():
        flattened_v = v.reshape((-1, v.shape[-1]))
        pmapped_and_vmapped_metrics[k] = wandb.Histogram(flattened_v)
        mean_metrics[f"{k}_mean"] = jnp.mean(flattened_v)
        std_metrics[f"{k}_std"] = jnp.std(flattened_v)

    return {**pmapped_and_vmapped_metrics, **mean_metrics, **std_metrics}


if __name__ == "__main__":
    pass
