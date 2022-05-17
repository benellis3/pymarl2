import copy
import os

import torch
import wandb
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from data.loader import features_to_mask_map

BATCH_DIM = 0
STEP_DIM = 1


class SingleMaskEarlyStopper:
    def __init__(self, early_stopper_patience, early_stopper_min_delta):
        self.patience = int(early_stopper_patience)
        self.min_delta = float(early_stopper_min_delta)
        self.waited_for = 0
        self.best_rmse = float('inf')
        self.best_metrics = None
        self.best_model = None
        self.best_epoch = 0

    def should_stop(self, metrics, model, epoch):
        rmse = metrics["rmse"]
        if rmse < self.best_rmse - self.min_delta:
            self.best_rmse = rmse
            self.best_metrics = metrics
            self.best_model = copy.deepcopy(model)
            self.best_epoch = epoch
            self.waited_for = 0
        else:
            self.waited_for += 1

        return self.waited_for > self.patience

    def save_best(self, save_dir):
        save_dir = save_dir / self.best_metrics["mask_name"]
        os.makedirs(save_dir, exist_ok=True)
        self.best_model.save_models(save_dir)
        metrics = pd.DataFrame(data={key: [val] for key, val in self.best_metrics.items()})
        metrics.to_json(save_dir / "metrics_dataframe.json")


class AllMasksEarlyStopper:
    def __init__(self, early_stopper_patience, early_stopper_min_delta):
        self.stoppers = dict([
            (mask_name, SingleMaskEarlyStopper(early_stopper_patience, early_stopper_min_delta))
            for mask_name in features_to_mask_map.keys()
        ])

    def should_stop(self, metrics_dict, epoch):
        _should_stop = True
        for mask_name, metrics in metrics_dict.items():
            _should_stop = _should_stop and self.stoppers[mask_name].should_stop(metrics, epoch)
        return _should_stop


def plot_line(x, y, title):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    return plt


def plot_line_wandb(x, y_label, title):
    data = [[x, y] for (x, y) in zip(range(len(x)), x)]
    table = wandb.Table(data=data, columns=["step", y_label])
    return wandb.plot.line(table, "step", y_label, title=title)


def mean_over_timestep(metric, is_step_mask, func=lambda x: x):
    step_metric = metric.squeeze(-1).sum(BATCH_DIM) / is_step_mask.squeeze(-1).sum(BATCH_DIM)
    return func(step_metric)


def plot_step_metric(step_metric):
    return plot_line(torch.arange(step_metric.size(0)).cpu().numpy(), step_metric.cpu().numpy(), '')


def plot_step_metric_v2(step_metric, title):
    return plot_line(np.arange(len(step_metric)), step_metric, title)


def preprocess_evaluation(target_model, loader, device):
    # Get tensor sizes.
    episode_sample = loader.episode_buffer[0]
    episode_sample.to(device)
    _, is_step_mask = target_model.forward(episode_sample)
    nb_steps = is_step_mask.size(-2)

    # Precompute metrics.
    max_t_filled = torch.tensor(0, device=device)
    max_target_q = torch.tensor(-float('inf'), device=device)
    min_target_q = torch.tensor(float('inf'), device=device)
    sum_target_q = torch.tensor(0., device=device)
    step_max_target_q = torch.ones((nb_steps,), device=device) * -float('inf')
    step_min_target_q = torch.ones((nb_steps,), device=device) * float('inf')
    step_sum_target_q = torch.zeros((nb_steps,), device=device)
    episodes_still_running = torch.zeros((nb_steps,), device=device)

    for episode_sample in loader.get_validation_batches(device):
        target_q_vals, is_step_mask = target_model.forward(episode_sample)
        max_t_filled = max(max_t_filled, episode_sample.max_t_filled())
        max_target_q = max(max_target_q, target_q_vals.max())
        min_target_q = min(min_target_q, target_q_vals.min())
        sum_target_q += target_q_vals.sum()
        step_max_target_q = torch.cat((step_max_target_q.unsqueeze(0), target_q_vals.squeeze(-1)), dim=0).max(dim=0)[0]
        step_min_target_q = torch.cat((step_min_target_q.unsqueeze(0), target_q_vals.squeeze(-1)), dim=0).min(dim=0)[0]
        step_sum_target_q += target_q_vals.squeeze(-1).sum(dim=BATCH_DIM)
        episodes_still_running += is_step_mask.squeeze(-1).sum(dim=BATCH_DIM)

    max_t_filled -= 1
    matrix_size = target_q_vals[:, :max_t_filled].size()
    step_max_target_q = step_max_target_q[:max_t_filled]
    step_min_target_q = step_min_target_q[:max_t_filled]
    step_sum_target_q = step_sum_target_q[:max_t_filled]
    episodes_still_running = episodes_still_running[:max_t_filled]

    total_steps = episodes_still_running.sum()
    range_target_q = max_target_q - min_target_q
    step_range_target_q = step_max_target_q - step_min_target_q
    mean_target_q = sum_target_q / total_steps
    step_mean_target_q = step_sum_target_q / episodes_still_running

    wandb.log({
        "max_t_filled": max_t_filled.item(),
        "max_target_q": max_target_q.item(),
        "min_target_q": min_target_q.item(),
        "mean_target_q": mean_target_q.item(),
        "range_target_q": range_target_q.item(),
        "step_max_target_q": plot_line_wandb(step_max_target_q.cpu().numpy(), "max_target_q", "step_max_target_q"),
        "step_min_target_q": plot_line_wandb(step_min_target_q.cpu().numpy(), "min_target_q", "step_min_target_q"),
        "step_mean_target_q": plot_line_wandb(step_mean_target_q.cpu().numpy(), "mean_target_q", "step_mean_target_q"),
        "step_range_target_q": plot_line_wandb(step_range_target_q.cpu().numpy(), "range_target_q",
                                               "step_range_target_q"),
        "episodes_still_running": plot_line_wandb(episodes_still_running.cpu().numpy(), "episodes",
                                                  "episodes_still_running"),
    })

    return (
        total_steps,
        max_t_filled,
        max_target_q,
        min_target_q,
        mean_target_q,
        step_max_target_q,
        step_min_target_q,
        step_mean_target_q,
        episodes_still_running,
        range_target_q,
        step_range_target_q,
        matrix_size,
    )


def evaluate_mask(eval_preprocess, epoch, model, target_model, loader, mask, criterion, device, mask_name):
    (
        total_steps, max_t_filled, max_target_q, min_target_q, mean_target_q,
        step_max_target_q, step_min_target_q, step_mean_target_q, episodes_still_running,
        range_target_q, step_range_target_q, matrix_size
    ) = eval_preprocess

    max_q = torch.tensor(-float('inf'), device=device)
    min_q = torch.tensor(float('inf'), device=device)
    sum_q = torch.tensor(0., device=device)
    step_max_q = torch.ones_like(step_max_target_q) * -float('inf')
    step_min_q = torch.ones_like(step_min_target_q) * float('inf')
    step_sum_q = torch.zeros_like(step_mean_target_q)

    loss = torch.zeros(matrix_size, device=device)
    error = torch.zeros(matrix_size, device=device)

    for episode_sample in loader.get_validation_batches(device):
        episode_sample = episode_sample[:, :max_t_filled + 1]
        target_q_vals, _ = target_model.forward(episode_sample)

        mask.apply(episode_sample, loader.state_feature_names, loader.obs_feature_names)
        q_vals, _ = model.forward(episode_sample)

        max_q = max(max_q, q_vals.max())
        min_q = min(min_q, q_vals.min())
        sum_q += q_vals.sum()
        step_max_q = torch.cat((step_max_q.unsqueeze(0), q_vals.squeeze(-1)), dim=0).max(dim=0)[0]
        step_min_q = torch.cat((step_min_q.unsqueeze(0), q_vals.squeeze(-1)), dim=0).min(dim=0)[0]
        step_sum_q += q_vals.squeeze(-1).sum(dim=BATCH_DIM)

        # Matrix metrics
        loss += criterion(q_vals, target_q_vals)
        error += torch.abs(q_vals - target_q_vals)

    # Aggregate.
    range_q = max_q - min_q
    step_range_q = step_max_q - step_min_q
    mean_q = sum_q / total_steps
    step_mean_q = step_sum_q / episodes_still_running

    loss = loss.sum() / total_steps
    rmse = torch.sqrt(loss)
    step_rmse = torch.sqrt(loss.squeeze(-1).sum(BATCH_DIM) / episodes_still_running)
    mean_error = error.sum() / total_steps
    step_mean_error = error.squeeze(-1).sum(BATCH_DIM) / episodes_still_running
    mean_deviation = mean_error / range_target_q * 100
    step_mean_deviation = step_mean_error / step_range_target_q * 100

    return {
        "max_q": max_q.item(),
        "min_q": min_q.item(),
        "mean_q": mean_q.item(),
        "range_q": range_q.item(),
        "step_max_q": step_max_q.cpu().numpy(),
        "step_min_q": step_min_q.cpu().numpy(),
        "step_mean_q": step_mean_q.cpu().numpy(),
        "episodes_still_running": episodes_still_running.cpu().numpy(),
        "step_range_q": step_range_q.cpu().numpy(),
        "loss:": loss.item(),
        "rmse": rmse.item(),
        "step_rmse": step_rmse.cpu().numpy(),
        "mean_error": mean_error.item(),
        "step_mean_error": step_mean_error.cpu().numpy(),
        "mean_deviation": mean_deviation.item(),
        "step_mean_deviation": step_mean_deviation.cpu().numpy(),
        "epoch": epoch,
        "mask_name": mask_name,
        "map_name": loader.map_name,
    }


def evaluate_single_mask(eval_preprocess, epoch, model, target_model, loader, mask, criterion, device, mask_name):
    with torch.no_grad():
        metrics = evaluate_mask(eval_preprocess, epoch, model, target_model, loader, mask, criterion, device, mask_name)
        metrics["mask_name"] = mask_name
        wandb.log({
            "validation/max_q": metrics["max_q"],
            "validation/min_q": metrics["min_q"],
            "validation/mean_q": metrics["mean_q"],
            "validation/range_q": metrics["range_q"],
            "validation/step_max_q": wandb.Image(plot_step_metric_v2(metrics["step_max_q"], mask_name)),
            "validation/step_min_q": wandb.Image(plot_step_metric_v2(metrics["step_min_q"], mask_name)),
            "validation/step_mean_q": wandb.Image(plot_step_metric_v2(metrics["step_mean_q"], mask_name)),
            "validation/episodes_still_running": wandb.Image(
                plot_step_metric_v2(metrics["episodes_still_running"], mask_name)),
            "validation/step_range_q": wandb.Image(plot_step_metric_v2(metrics["step_range_q"], mask_name)),
            "validation/rmse": metrics["rmse"],
            "validation/step_rmse": wandb.Image(plot_step_metric_v2(metrics["step_rmse"], mask_name)),
            "validation/step_rmse_line": plot_line_wandb(metrics["step_rmse"], "step_rmse", "rmse"),
            "validation/mean_error": metrics["mean_error"],
            "validation/step_mean_error": wandb.Image(plot_step_metric_v2(metrics["step_mean_error"], mask_name)),
            "validation/mean_deviation": metrics["mean_deviation"],
            "validation/step_mean_deviation": wandb.Image(
                plot_step_metric_v2(metrics["step_mean_deviation"], mask_name)),
            "epoch": epoch,
        }, step=epoch)
        plt.close("all")
        return metrics


# TODO: WIP

def evaluate_all_masks(epoch, model, target_model, loader, criterion, device):
    new_eval_metrics = {}
    for mask_name, mask in features_to_mask_map.items():
        metric = evaluate_single_mask(epoch, model, target_model, loader, mask, criterion, device, mask_name)
        new_eval_metrics[mask_name] = metric

    return new_eval_metrics


def evaluate_all_masks_v2(epoch, model, target_model, loader, criterion, device):
    def eval_(inpu):
        mask_name, mask = inpu
        metric = evaluate_single_mask(epoch, model, target_model, loader, mask, criterion, device, mask_name)
        return mask_name, metric

    new_eval_metrics = {}
    evals = map(eval_, features_to_mask_map.items())
    for mask_name, metric in evals:
        new_eval_metrics[mask_name] = metric

    return new_eval_metrics
