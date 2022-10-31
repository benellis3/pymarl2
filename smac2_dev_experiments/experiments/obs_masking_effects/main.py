import pathlib
import wandb
import torch
import torch.nn as nn
from tqdm import tqdm

from data.loader import Loader, build_mask
from models.nqmix import NQmixNet
from experiments.obs_masking_effects.train import train, preprocess_training
from experiments.obs_masking_effects.evaluate import SingleMaskEarlyStopper, \
    evaluate_single_mask, preprocess_evaluation, log_best_results
from experiments.obs_masking_effects.default_params import parse_arguments


def get_save_dir_prefix(config):
    save_dir = pathlib.Path(__file__).parent.resolve()
    save_dir = save_dir / "results" / f"smac_{config.smac_version}" / f"{config.unit_map_name}" / str(config.seed)
    return save_dir


# Set experiment parameter.
params = parse_arguments()
params.unit_map_name = f"{params.nb_units}_{params.map_name}" if int(params.smac_version) == 2 else params.map_name

# Wandb updates the config if the script is running in a sweeping agent.
run = wandb.init(project=params.project,
                 entity=params.entity,
                 config=vars(params),
                 group=params.group,
                 name=f"{params.mask_name}_{params.unit_map_name}",
                 mode="online")
config = wandb.config  # Updated from wandb server if running a sweep.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, config)

# Training artifacts.
training_loader = Loader(map_name=config.unit_map_name, dataset_type='train', batch_size=config.batch_size,
                         smac_version=int(config.smac_version), seed=int(config.seed))
evaluation_loader = Loader(map_name=config.unit_map_name, dataset_type='evaluate', batch_size=config.batch_size,
                           smac_version=int(config.smac_version), seed=int(config.seed))

model = NQmixNet(training_loader, device)
target_model = NQmixNet(training_loader, device, is_target=True)

mask = build_mask(config.mask_name)
criterion = nn.MSELoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
early_stopper = SingleMaskEarlyStopper(config.early_stopper_patience, config.early_stopper_min_delta)

# Training.
preprocess_training(training_loader)
eval_preprocess = preprocess_evaluation(target_model, evaluation_loader, device)
for epoch in tqdm(range(1, 1 + config.epochs)):
    train(epoch, model, target_model, training_loader, mask, criterion, optimizer, device)

    if epoch % config.eval_every == 0:
        eval_metrics = evaluate_single_mask(eval_preprocess, epoch, model, target_model, evaluation_loader, mask,
                                            criterion, device, config.mask_name)
        if early_stopper.should_stop(eval_metrics, model, epoch):
            print(f"Early stopping at epoch {epoch}. Best achieved at {early_stopper.best_epoch}.")
            break

log_best_results(early_stopper.best_metrics)
if config.save_best:
    early_stopper.save_best(get_save_dir_prefix(config))
