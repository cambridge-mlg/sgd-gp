from pathlib import Path

import jax
import jax.random as jr
import ml_collections.config_flags
import wandb
from absl import app, flags
from data import get_dataset
from eval_utils import RMSE
from kernels import RBFKernel
from linear_model import marginal_likelihood
from models import ExactGPModel
from utils import ExactMetricsTuple, HparamsTuple, flatten_nested_dict, setup_training, update_config_dict

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

        kernel = RBFKernel(config.kernel_config)

        save_dir = Path(config.save_dir).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)

        key = jr.PRNGKey(config.seed)
        optim_key, sampling_key, key = jr.split(key, 3)

        # Compute exact solution
        exact_model, exact_metrics = None, None
        if config.compute_exact_soln:
            exact_model = ExactGPModel(config.dataset_config.noise_scale, kernel)
            exact_model.compute_representer_weights(train_ds)
            y_pred_exact = exact_model.predictive_mean(train_ds, test_ds)
            test_rmse_exact = RMSE(test_ds.y, y_pred_exact, mu=train_ds.mu_y, sigma=train_ds.sigma_y)

            print(f"test_rmse_exact = {test_rmse_exact}")
            wandb.log({"test_rmse_exact": test_rmse_exact})
            # Define exact metrics that we will use later to compare with stochastic solution
            ExactMetricsTuple(
                alpha=exact_model.alpha,
                y_pred=y_pred_exact,
                test_rmse=test_rmse_exact
            )
            
            # Optimise Marginal Likelihood
            init_hparams = HparamsTuple(
                length_scale=exact_model.kernel.kernel_config["length_scale"],
                signal_scale=exact_model.kernel.kernel_config["signal_scale"],
                noise_scale=exact_model.noise_scale,)
            
            print("initial mll : ", marginal_likelihood(train_ds.x, train_ds.y, kernel.kernel_fn, init_hparams))
            mll_config = ml_collections.ConfigDict()
            mll_config.learning_rate = 0.1
            mll_config.iterations = 100
            hparams = exact_model.compute_mll_optim(init_hparams, train_ds, mll_config, test_ds)
            
            print(hparams)
            # TODO: implement helper that updates hparams for GPModel class.

        # Compute stochastic optimised solution
        # model = SGDGPModel(config.dataset_config.noise_scale, kernel)

        # metrics_list = ["loss", "test_rmse", "err", "reg"]
        # if config.compute_exact_soln:
        #     metrics_list.extend(["alpha_diff", "y_pred_diff", "test_rmse_diff", "alpha_rkhs_diff"])

        # # Compute the SGD MAP solution for representer weights.
        # model.compute_representer_weights(
        #     optim_key,
        #     train_ds,
        #     test_ds,
        #     config.train_config,
        #     metrics_list=metrics_list,
        #     metrics_prefix="train",
        #     exact_metrics=exact_metrics if config.compute_exact_soln else None,
        # )
        
        # zero_mean_samples, alpha_samples = model.compute_posterior_samples(
        #     sampling_key,
        #     n_samples=config.sampling_config.n_samples,
        #     train_ds=train_ds,
        #     test_ds=test_ds,
        #     config=config.sampling_config,
        #     use_rff=False,
        #     n_features=config.sampling_config.n_features_optim,
        #     zero_mean=True,
        #     metrics_list=metrics_list,
        #     metrics_prefix="sampling",
        #     compare_exact=True
        # )

        # return zero_mean_samples, alpha_samples


if __name__ == "__main__":
    # Adds jax flags to the program.
    jax.config.config_with_absl()

    # Snippet to pass configs into the main function.
    def _main(argv):
        del argv
        config = FLAGS.config
        main(config)

    app.run(_main)
