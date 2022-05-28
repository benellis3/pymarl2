import copy
import os

import torch
import wandb
import pandas as pd

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


def plot_line_wandb(y, y_label, title):
    data = [[x, y] for (x, y) in enumerate(y)]
    table = wandb.Table(data=data, columns=["step", y_label])
    return wandb.plot.line(table, "step", y_label, title=title)


def preprocess_evaluation(target_model, loader, device):
    with torch.no_grad():
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
            step_max_target_q = \
                torch.cat((step_max_target_q.unsqueeze(0), target_q_vals.squeeze(-1)), dim=0).max(dim=0)[0]
            step_min_target_q = \
                torch.cat((step_min_target_q.unsqueeze(0), target_q_vals.squeeze(-1)), dim=0).min(dim=0)[0]
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
            "target/max_t_filled": max_t_filled.item(),
            "target/max_target_q": max_target_q.item(),
            "target/min_target_q": min_target_q.item(),
            "target/mean_target_q": mean_target_q.item(),
            "target/range_target_q": range_target_q.item(),
            "target/step_max_target_q": plot_line_wandb(step_max_target_q.cpu().numpy(), "max_target_q",
                                                        "step_max_target_q"),
            "target/step_min_target_q": plot_line_wandb(step_min_target_q.cpu().numpy(), "min_target_q",
                                                        "step_min_target_q"),
            "target/step_mean_target_q": plot_line_wandb(step_mean_target_q.cpu().numpy(), "mean_target_q",
                                                         "step_mean_target_q"),
            "target/step_range_target_q": plot_line_wandb(step_range_target_q.cpu().numpy(), "range_target_q",
                                                          "step_range_target_q"),
            "target/episodes_still_running": plot_line_wandb(episodes_still_running.cpu().numpy(), "episodes",
                                                             "episodes_still_running"),
            "validation/batch/batch_size": loader.batch_size,
            "validation/batch/batch_in_epoch": loader.nb_batch
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
    q_deviation = torch.zeros(matrix_size, device=device)

    for episode_sample in loader.get_validation_batches(device, max_seq_length=max_t_filled + 1):
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
        q_deviation += (error / torch.abs(target_q_vals)).nan_to_num()

    # Aggregate.
    range_q = max_q - min_q
    step_range_q = step_max_q - step_min_q
    mean_q = sum_q / total_steps
    step_mean_q = step_sum_q / episodes_still_running

    mean_loss = loss.sum() / total_steps
    rmse = torch.sqrt(mean_loss)
    step_rmse = torch.sqrt(loss.squeeze(-1).sum(BATCH_DIM) / episodes_still_running)

    rmse_by_q = rmse / mean_q * 100
    step_rmse_by_q = step_rmse / step_mean_q * 100

    mean_error = error.sum() / total_steps
    step_mean_error = error.squeeze(-1).sum(BATCH_DIM) / episodes_still_running

    mean_q_deviation = q_deviation.sum() / total_steps * 100
    step_mean_q_deviation = q_deviation.squeeze(-1).sum(BATCH_DIM) / episodes_still_running * 100

    mean_range_deviation = mean_error / range_target_q * 100
    step_mean_range_deviation = step_mean_error / step_range_target_q * 100

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
        "mean_loss:": mean_loss.item(),
        "rmse": rmse.item(),
        "step_rmse": step_rmse.cpu().numpy(),
        "rmse_by_q": rmse_by_q.item(),
        "step_rmse_by_q": step_rmse_by_q.cpu().numpy(),
        "mean_error": mean_error.item(),
        "step_mean_error": step_mean_error.cpu().numpy(),
        "mean_q_deviation": mean_q_deviation.item(),
        "step_mean_q_deviation": step_mean_q_deviation.cpu().numpy(),
        "mean_range_deviation": mean_range_deviation.item(),
        "step_mean_range_deviation": step_mean_range_deviation.cpu().numpy(),
        "epoch": epoch,
        "mask_name": mask_name,
        "map_name": loader.map_name,
    }


def evaluate_single_mask(eval_preprocess, epoch, model, target_model, loader, mask, criterion, device, mask_name):
    with torch.no_grad():
        metrics = evaluate_mask(eval_preprocess, epoch, model, target_model, loader, mask, criterion, device, mask_name)
        wandb.log({
            "validation/q/max_q": metrics["max_q"],
            "validation/q/min_q": metrics["min_q"],
            "validation/q/mean_q": metrics["mean_q"],
            "validation/q/range_q": metrics["range_q"],
            "validation/q/step_max_q": plot_line_wandb(metrics["step_max_q"], "max_q", "step_max_q"),
            "validation/q/step_min_q": plot_line_wandb(metrics["step_min_q"], "min_q", "step_min_q"),
            "validation/q/step_mean_q": plot_line_wandb(metrics["step_mean_q"], "mean_q", "step_mean_q"),
            "validation/q/step_range_q": plot_line_wandb(metrics["step_range_q"], "range_q", "step_range_q"),
            "validation/step/episodes_still_running": plot_line_wandb(metrics["episodes_still_running"], "episodes",
                                                                      "episodes_still_running"),
            "validation/rmse": metrics["rmse"],
            "validation/step/step_rmse": plot_line_wandb(metrics["step_rmse"], "rmse", "step_rmse"),
            "validation/rmse_by_q": metrics["rmse_by_q"],
            "validation/step/step_rmse_by_q": plot_line_wandb(metrics["step_rmse_by_q"], "rmse_by_q",
                                                              "step_rmse_by_q"),
            "validation/mean_error": metrics["mean_error"],
            "validation/step/step_mean_error": plot_line_wandb(metrics["step_mean_error"], "mean_error",
                                                               "step_mean_error"),
            "validation/mean_q_deviation": metrics["mean_q_deviation"],
            "validation/step/step_mean_q_deviation": plot_line_wandb(metrics["step_mean_q_deviation"],
                                                                     "mean_q_deviation", "step_mean_q_deviation"),
            "validation/mean_range_deviation": metrics["mean_range_deviation"],
            "validation/step/step_mean_range_deviation": plot_line_wandb(metrics["step_mean_range_deviation"],
                                                                         "mean_range_deviation",
                                                                         "step_mean_range_deviation"),
            "epoch": epoch,
        }, step=epoch)
        return metrics


def log_best_results(metrics):
    wandb.log({
        "validation/best/q/max_q": metrics["max_q"],
        "validation/best/q/min_q": metrics["min_q"],
        "validation/best/q/mean_q": metrics["mean_q"],
        "validation/best/q/range_q": metrics["range_q"],
        "validation/best/q/step_max_q": plot_line_wandb(metrics["step_max_q"], "max_q", "step_max_q"),
        "validation/best/q/step_min_q": plot_line_wandb(metrics["step_min_q"], "min_q", "step_min_q"),
        "validation/best/q/step_mean_q": plot_line_wandb(metrics["step_mean_q"], "mean_q", "step_mean_q"),
        "validation/best/q/step_range_q": plot_line_wandb(metrics["step_range_q"], "range_q", "step_range_q"),
        "validation/best/step/episodes_still_running": plot_line_wandb(metrics["episodes_still_running"], "episodes",
                                                                       "episodes_still_running"),
        "validation/best/rmse": metrics["rmse"],
        "validation/best/step/step_rmse": plot_line_wandb(metrics["step_rmse"], "rmse", "step_rmse"),
        "validation/best/rmse_by_q": metrics["rmse_by_q"],
        "validation/best/step/step_rmse_by_q": plot_line_wandb(metrics["step_rmse_by_q"], "rmse_by_q",
                                                               "step_rmse_by_q"),
        "validation/best/mean_error": metrics["mean_error"],
        "validation/best/step/step_mean_error": plot_line_wandb(metrics["step_mean_error"], "mean_error",
                                                                "step_mean_error"),
        "validation/best/mean_q_deviation": metrics["mean_q_deviation"],
        "validation/best/step/step_mean_q_deviation": plot_line_wandb(metrics["step_mean_q_deviation"],
                                                                      "mean_q_deviation", "step_mean_q_deviation"),
        "validation/best/mean_range_deviation": metrics["mean_range_deviation"],
        "validation/best/step/step_mean_range_deviation": plot_line_wandb(metrics["step_mean_range_deviation"],
                                                                          "mean_range_deviation",
                                                                          "step_mean_range_deviation"),
        "validation/best/epoch": metrics["epoch"],
    })
