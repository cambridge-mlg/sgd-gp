import itertools
import subprocess
from concurrent.futures import ThreadPoolExecutor


def run_experiment(lr, batch_size):
    dataset_name = '3droad'
    tpu_name = f"tpu-hparamsweep-{dataset_name}-{lr}-{batch_size}"

    # Create the TPU
    create_tpu_command = [
        "gcloud", "compute", "tpus", "tpu-vm", "create", tpu_name,
        "--zone", "us-central1-f",
        "--accelerator-type", "v2-8",
        "--version", "tpu-vm-base",
        "--preemptible"
    ]
    with open(f"logs/create_tpu_hparamsweep_{dataset_name}_{lr}_{batch_size}.log", "w") as create_tpu_log_file:
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
                     f"python scripts/obtain_mean.py 9835d6db89010f73306f92bb9a080c9751b25d28 --config configs/default.py:{dataset_name} --config.model_name sgd --config.wandb.log --config.wandb.name {dataset_name}_hparamsweep_lr={lr}_bs={batch_size} --config.train_config.learning_rate {lr} --config.train_config.batch_size {batch_size}",
        "--", "-t"
    ]
    with open(f"logs/run_experiment_hparamsweep_{dataset_name}_{lr}_{batch_size}.log", "w") as run_experiment_log_file:
        subprocess.run(run_experiment_command, check=True, stdout=run_experiment_log_file, stderr=subprocess.STDOUT)

    # Delete the TPU
    delete_tpu_command = [
        "gcloud", "compute", "tpus", "tpu-vm", "delete", tpu_name,
        "--zone", "us-central1-f",
        "--quiet"
    ]
    with open(f"logs/delete_tpu_hparamsweep_{dataset_name}_{lr}_{batch_size}.log", "w") as delete_tpu_log_file:
        subprocess.run(delete_tpu_command, check=True, stdout=delete_tpu_log_file, stderr=subprocess.STDOUT)

lrs = [0.5, 1., 2., 5.]
batch_sizes = [256, 512, 1024]

# Run the experiments in parallel using a ThreadPoolExecutor
with ThreadPoolExecutor() as executor:

    tasks = itertools.product(lrs, batch_sizes)
    executor.map(lambda task: run_experiment(*task), tasks)