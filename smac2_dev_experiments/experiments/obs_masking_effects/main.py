import pathlib
import os
import wandb
import torch
import torch.nn as nn
from tqdm import tqdm

from data.loader import Loader, build_mask
from models.nqmix import NQmixNet
from experiments.obs_masking_effects.train import train_single_transform
from experiments.obs_masking_effects.evaluate import evaluate_single_transform
from experiments.obs_masking_effects.default_params import parse_arguments


def save_model(model, epoch, config):
    save_dir = pathlib.Path(__file__).parent.resolve()
    save_dir = save_dir / "results" / config.map_name / config.launch_time / f"epoch_{epoch}"
    os.makedirs(save_dir, exist_ok=True)
    model.save_models(save_dir)


# Set experiment parameter.
params = parse_arguments()
# Wandb updates the config if the script is running in a sweeping agent.
run = wandb.init(project=params.project,
                 entity=params.entity,
                 config=vars(params),
                 group=params.group,
                 mode="online")
config = wandb.config  # Updated from wandb server if running a sweep.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, config)

# Training artifacts.
training_loader = Loader(map_name=config.map_name, dataset_type='train', batch_size=config.batch_size)
evaluation_loader = Loader(map_name=config.map_name, dataset_type='evaluate')

model = NQmixNet(training_loader, device)
target_model = NQmixNet(training_loader, device, is_target=True)

mask_transform = build_mask(config.mask_name, mask_state=bool(int(config.mask_state)))

criterion = nn.MSELoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# Training.
for epoch in tqdm(range(1, 1 + config.epochs)):
    train_single_transform(epoch, model, target_model, training_loader, mask_transform, criterion, optimizer, device)
    evaluate_single_transform(epoch, model, target_model, evaluation_loader, mask_transform, criterion, device)
    if config.save_models and epoch % int(config.save_every) == 0:
        save_model(model, epoch, config)
