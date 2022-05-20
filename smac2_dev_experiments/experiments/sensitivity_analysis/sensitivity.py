import numpy as np
import torch
import wandb

BATCH_DIM = 0
EPISODE_STEP_DIM = 1
AGENT_DIM = 2
FEATURE_DIM = -1


def plot_line_wandb(y, y_label, title):
    data = [[x, y] for (x, y) in enumerate(y)]
    table = wandb.Table(data=data, columns=["step", y_label])
    return wandb.plot.line(table, "step", y_label, title=title)


def plot_bar_wandb(metric, title):
    data = [[label, val] for (label, val) in metric.items()]
    table = wandb.Table(data=data, columns=["feature", "gradient"])
    return wandb.plot.bar(table, "feature", "gradient", title=title)


def preprocess_sensitivity(target_model, loader, device):
    episode_sample = loader.episode_buffer[0]
    episode_sample.to(device)
    q_vals, is_step_mask = target_model.forward(episode_sample)
    nb_steps = is_step_mask.size(-2)

    max_t_filled = torch.tensor(0, device=device)
    episodes_still_running = torch.zeros((nb_steps,), device=device)

    with torch.no_grad():
        for episode_sample in loader.get_validation_batches(device):
            _, is_step_mask = target_model.forward(episode_sample)
            max_t_filled = max(max_t_filled, episode_sample.max_t_filled())
            episodes_still_running += is_step_mask.squeeze(-1).sum(dim=BATCH_DIM)

    max_t_filled -= 1
    obs_grad_matrix_size = (max_t_filled.item(), episode_sample["obs"].size(-1))
    state_grad_matrix_size = (max_t_filled.item(), episode_sample["state"].size(-1))
    episodes_still_running = episodes_still_running[:max_t_filled]
    total_steps = episodes_still_running.sum()

    return total_steps, max_t_filled, episodes_still_running, obs_grad_matrix_size, state_grad_matrix_size


def compute_sensitivity(target_model, loader, device):
    (
        total_steps, max_t_filled, episodes_still_running, obs_grad_matrix_size, state_grad_matrix_size
    ) = preprocess_sensitivity(target_model, loader, device)

    obs_grad = torch.zeros(obs_grad_matrix_size, device=device)
    state_grad = torch.zeros(state_grad_matrix_size, device=device)

    for episode_sample in loader.get_validation_batches(device, max_seq_length=max_t_filled + 1):
        episode_sample["obs"].requires_grad_(True)
        episode_sample["state"].requires_grad_(True)

        q_vals, is_step_mask = target_model.forward(episode_sample)
        q_vals.sum().backward()

        obs_grad += episode_sample["obs"].grad[:, :-1].abs().mean(AGENT_DIM).sum(BATCH_DIM)
        state_grad += episode_sample["state"].grad[:, :-1].abs().sum(BATCH_DIM)

    obs_grad = obs_grad / episodes_still_running.unsqueeze(-1)
    state_grad = state_grad / episodes_still_running.unsqueeze(-1)
    mean_obs_grad = obs_grad.mean(BATCH_DIM)
    mean_state_grad = state_grad.mean(BATCH_DIM)

    state_grads = {
        feature_name: state_grad[:, i].cpu().numpy() for i, feature_name in enumerate(loader.state_feature_names)
    }
    mean_state_grads = {
        feature_name: mean_state_grad[i].item() for i, feature_name in enumerate(loader.state_feature_names)
    }
    obs_grads = {
        feature_name: obs_grad[:, i].cpu().numpy() for i, feature_name in enumerate(loader.obs_feature_names)
    }
    mean_obs_grads = {
        feature_name: mean_obs_grad[i].item() for i, feature_name in enumerate(loader.obs_feature_names)
    }

    wandb.log({
        "sensitivity/state": wandb.plot.line_series(
            xs=np.arange(max_t_filled.item()),
            ys=list(state_grads.values()),
            keys=list(state_grads.keys()),
            title="State Feature Sensitivity",
            xname="episode step"),
        "sensitivity/observations": wandb.plot.line_series(
            xs=np.arange(max_t_filled.item()),
            ys=list(obs_grads.values()),
            keys=list(obs_grads.keys()),
            title="Observations Feature Sensitivity",
            xname="episode step"),
        "sensitivity/mean_state": plot_bar_wandb(mean_state_grads, "Mean State Sensitivity"),
        "sensitivity/mean_observations": plot_bar_wandb(mean_obs_grads, "Mean Observations Sensitivity"),
        "sensitivity/episodes_still_running": plot_line_wandb(episodes_still_running.cpu().numpy(), "episodes",
                                                              "episodes_still_running"),
        "sensitivity/null": 0,
    })
    return {
        "state": state_grads,
        "observations": obs_grads,
        "mean_state": mean_state_grads,
        "mean_observations": mean_obs_grads,
        "episodes_still_running": {"episodes_still_running": episodes_still_running.cpu().numpy()},
    }
