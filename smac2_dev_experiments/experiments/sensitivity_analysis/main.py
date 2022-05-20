import pathlib

import os

import pandas as pd
import torch
import wandb

from data.loader import Loader
from experiments.sensitivity_analysis.default_params import parse_arguments

# Set experiment parameter.
from experiments.sensitivity_analysis.sensitivity import compute_sensitivity
from models.nqmix import NQmixNet


def save_results(results, config):
    save_dir = pathlib.Path(__file__).parent.resolve()
    save_dir = save_dir / "results" / config.map_name
    os.makedirs(save_dir, exist_ok=True)
    for metric_name, metric_value in results.items():
        metric = pd.DataFrame(data={key: [val] for key, val in metric_value.items()})
        metric.to_json(save_dir / f"{metric_name}.json")


params = parse_arguments()
# Wandb updates the config if the script is running in a sweeping agent.
run = wandb.init(project=params.project,
                 entity=params.entity,
                 config=vars(params),
                 group=params.group,
                 name=f"{params.map_name}_{params.launch_time}",
                 mode="online")
config = wandb.config  # Updated from wandb server if running a sweep.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, config)

data_loader = Loader(map_name=config.map_name, dataset_type='evaluate', batch_size=128)
target_model = NQmixNet(data_loader, device, is_target=True, sensitivity=True)

results = compute_sensitivity(target_model, data_loader, device)
save_results(results, config)
