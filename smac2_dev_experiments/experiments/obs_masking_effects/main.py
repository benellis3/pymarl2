import pathlib
import os
import wandb
import torch
import torch.nn as nn
from tqdm import tqdm

from data.loader import Loader, build_mask
from models.nqmix import NQmixNet
from experiments.obs_masking_effects.train import train
from experiments.obs_masking_effects.evaluate import SingleMaskEarlyStopper, AllMasksEarlyStopper, evaluate_single_mask, \
    evaluate_all_masks, preprocess_evaluation
from experiments.obs_masking_effects.default_params import parse_arguments


def get_save_dir_prefix(config):
    save_dir = pathlib.Path(__file__).parent.resolve()
    save_dir = save_dir / "best_results" / config.map_name
    return save_dir

    model.save_models(save_dir)


# Set experiment parameter.
params = parse_arguments()
# Wandb updates the config if the script is running in a sweeping agent.
run = wandb.init(project=params.project,
                 entity=params.entity,
                 config=vars(params),
                 group=params.group,
                 mode="disabled")
config = wandb.config  # Updated from wandb server if running a sweep.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, config)

# Training artifacts.
training_loader = Loader(map_name=config.map_name, dataset_type='train', batch_size=config.batch_size)
evaluation_loader = Loader(map_name=config.map_name, dataset_type='evaluate')

model = NQmixNet(training_loader, device)
target_model = NQmixNet(training_loader, device, is_target=True)

mask = build_mask(config.mask_name)
criterion = nn.MSELoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
Stopper = AllMasksEarlyStopper if config.mask_name == "random" else SingleMaskEarlyStopper
early_stopper = Stopper(config.early_stopper_patience, config.early_stopper_min_delta)

# Training.
eval_preprocess = preprocess_evaluation(target_model, evaluation_loader, device)
for epoch in tqdm(range(1, 1 + config.epochs)):
    train(epoch, model, target_model, training_loader, mask, criterion, optimizer, device)

    if epoch % config.eval_every == 0:
        if config.mask_name == "random":
            eval_metrics = evaluate_all_masks(epoch, model, target_model, evaluation_loader, criterion, device)
        else:
            eval_metrics = evaluate_single_mask(eval_preprocess, epoch, model, target_model, evaluation_loader, mask,
                                                criterion, device, config.mask_name)

        if early_stopper.should_stop(eval_metrics, model, epoch):
            print(f"Early stopping at epoch {epoch}")
            if config.save_best:
                early_stopper.save_best(get_save_dir_prefix(config))
            break
