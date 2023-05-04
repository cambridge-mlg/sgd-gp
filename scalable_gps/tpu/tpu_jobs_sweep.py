import itertools
import subprocess
from concurrent.futures import ThreadPoolExecutor


def run_experiment(seed, dataset_name, method_name):
    tpu_name = f"tpu-{dataset_name}-{method_name}-{seed}"

    try:
        # Create the TPU
        create_tpu_command = [
            "gcloud", "compute", "tpus", "tpu-vm", "create", tpu_name,
            "--zone", "us-central1-f",
            "--accelerator-type", "v2-8",
            "--version", "tpu-vm-base",
            "--preemptible"
        ]
        with open(f"logs/create_tpu_{dataset_name}_{method_name}_{seed}.log", "w") as create_tpu_log_file:
            subprocess.run(create_tpu_command, check=True, stdout=create_tpu_log_file, stderr=subprocess.STDOUT)

        # Run the experiment
        run_experiment_command = [
            "gcloud", "compute", "tpus", "tpu-vm", "ssh", tpu_name, "--zone", "us-central1-f",
            "--command", "git clone https://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@github.com/jandylin/scalable-gaussian-processes.git; "
                        "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; "
                        "chmod +x Miniconda3-latest-Linux-x86_64.sh; "
                        "./Miniconda3-latest-Linux-x86_64.sh -b; "
                        'export PATH="$HOME/miniconda3/bin:$PATH"; '
                        "conda env create -f scalable-gaussian-processes/environment_tpu.yaml; "
                        "source /home/shreyaspadhy_gmail_com/miniconda3/etc/profile.d/conda.sh; "
                        "conda activate jax; "
                        "cd scalable-gaussian-processes/; "
                        "pip install -e .; "
                        "cd scalable_gps/; "
                        f'export PYTHONPATH="/path/to/scalable_gaussian_processes:$PYTHONPATH"; '
                        f"python scripts/obtain_mean.py 9835d6db89010f73306f92bb9a080c9751b25d28 --config configs/default.py:{dataset_name} --config.model_name {method_name} --config.dataset_config.split {seed} --config.wandb.log --config.wandb.name {dataset_name}_{method_name}_{seed} --config.wandb.log_artifact",
            "--", "-t"
        ]
        with open(f"logs/run_experiment_{dataset_name}_{method_name}_{seed}.log", "w") as run_experiment_log_file:
            subprocess.run(run_experiment_command, check=True, stdout=run_experiment_log_file, stderr=subprocess.STDOUT)
    except Exception as e:
        print(f"An error occurred while running the experiment for tpu-{dataset_name}-{method_name}-{seed}: {e}")
        raise

    finally:
        # Delete the TPU
        delete_tpu_command = [
            "gcloud", "compute", "tpus", "tpu-vm", "delete", tpu_name,
            "--zone", "us-central1-f",
            "--quiet"
        ]
        with open(f"logs/delete_tpu_{dataset_name}_{method_name}_{seed}.log", "w") as delete_tpu_log_file:
            subprocess.run(delete_tpu_command, check=True, stdout=delete_tpu_log_file, stderr=subprocess.STDOUT)


seeds = [0]
datasets = ['yacht'] # ['3droad']
methods = ['sgd', 'cg', 'precondcg']

# Run the experiments in parallel using a ThreadPoolExecutor
with ThreadPoolExecutor() as executor:

    tasks = itertools.product(seeds, datasets, methods)
    executor.map(lambda task: run_experiment(*task), tasks)