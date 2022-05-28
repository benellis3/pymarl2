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
    save_dir = save_dir / "results" / f"smac_{config.smac_version}" / f"{config.unit_map_name}" / str(config.seed)
    os.makedirs(save_dir, exist_ok=True)
    for metric_name, metric_value in results.items():
        metric = pd.DataFrame(data={key: [val] for key, val in metric_value.items()})
        metric.to_json(save_dir / f"{metric_name}.json")


params = parse_arguments()
params.unit_map_name = f"{params.nb_units}_{params.map_name}" if int(params.smac_version) == 2 else params.map_name
# Wandb updates the config if the script is running in a sweeping agent.
run = wandb.init(project=params.project,
                 entity=params.entity,
                 config=vars(params),
                 group=params.group,
                 name=f"sensitivity-{params.unit_map_name}",
                 mode="online")
config = wandb.config  # Updated from wandb server if running a sweep.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, config)

data_loader = Loader(map_name=config.map_name, dataset_type='evaluate',
                     smac_version=int(config.smac_version), seed=int(config.seed))
target_model = NQmixNet(data_loader, device, is_target=True, sensitivity=True)

results = compute_sensitivity(target_model, data_loader, device)
save_results(results, config)
