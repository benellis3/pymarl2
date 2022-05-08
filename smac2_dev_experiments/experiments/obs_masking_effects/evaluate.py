import torch
import wandb

import matplotlib.pyplot as plt

BATCH_DIM = 0
STEP_DIM = 1


def plot_line(x, y):
    plt.figure()
    plt.plot(x, y)
    return plt


def evaluate_all_transforms():
    pass


def mean_over_timestep(metric, is_step_mask, func=lambda x: x):
    step_metric = metric.squeeze(-1).sum(BATCH_DIM) / is_step_mask.squeeze(-1).sum(BATCH_DIM)
    return func(step_metric)


def median_over_timestep(metric, is_step_mask, func=lambda x: x):
    step_median_metric = torch.zeros(metric.size(STEP_DIM))
    for step in range(metric.size(STEP_DIM)):
        step_median_metric[step] = torch.median(
            torch.masked_select(metric[:, step].squeeze(-1),
                                is_step_mask[:, step].squeeze(-1) == 1)
        )
    return func(step_median_metric)


def plot_step_metric(step_metric):
    return plot_line(torch.arange(step_metric.size(0)).cpu().numpy(), step_metric.cpu().numpy())


def evaluate_single_transform(epoch, model, target_model, loader, mask_transform, criterion, device):
    with torch.no_grad():
        episode_sample = loader.sample_batch(device)
        target_q_vals, is_step_mask = target_model.forward(episode_sample)
        total_steps = is_step_mask.sum()

        mask_transform(episode_sample, loader.state_feature_names, loader.obs_feature_names)
        q_vals, _ = model.forward(episode_sample)

        # Grounding metrics.
        max_q = target_q_vals.max()
        min_q = target_q_vals.min()
        mean_q = (target_q_vals.sum() / total_steps)
        q_range = torch.abs(max_q - min_q)

        # Matrix metrics
        loss = criterion(q_vals, target_q_vals)
        error = torch.abs(q_vals - target_q_vals)
        deviation = (error / q_range) * 100
        deviation[~torch.isfinite(deviation)] = 0  # Steps that are not counted in the mean anyway.

        # Mean metrics
        mean_loss = loss.sum() / total_steps
        mean_error = error.sum() / total_steps
        mean_deviation = deviation.sum() / total_steps

        # Median metrics
        median_error = torch.median(torch.masked_select(error, is_step_mask == 1))

        # Mean over timestep metrics
        step_rmse = mean_over_timestep(loss, is_step_mask, func=torch.sqrt)
        step_mean_error = mean_over_timestep(error, is_step_mask)

        # Median over timestep metrics
        step_median_error = median_over_timestep(error, is_step_mask)

        wandb.log({
            "validation/loss": mean_loss.item(),

            "validation/rmse": torch.sqrt(mean_loss).item(),
            "validation/step_rmse": wandb.Image(plot_step_metric(step_rmse)),

            "validation/mean_error": mean_error.item(),
            "validation/median_error": median_error.item(),
            "validation/step_mean_error": wandb.Image(plot_step_metric(step_mean_error)),
            "validation/step_median_error": wandb.Image(plot_step_metric(step_median_error)),

            "validation/mean_deviation": mean_deviation.item(),

            "validation/max_target_q": max_q.item(),
            "validation/min_target_q": min_q.item(),
            "validation/mean_target_q": mean_q.item(),

            "validation/max_model_q": q_vals.max().item(),
            "validation/min_model_q": q_vals.min().item(),
            "validation/mean_model_q": (q_vals.sum() / total_steps).item(),

            "epoch": epoch,
            "validation/batch/size": loader.batch_size,
            "validation/batch/in_epoch": loader.nb_batch
        }, step=epoch)
        plt.close('all')
