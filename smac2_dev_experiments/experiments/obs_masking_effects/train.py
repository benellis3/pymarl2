import wandb
import torch


def train(epoch, model, target_model, loader, mask, criterion, optimizer, device):
    running_loss = torch.tensor(0., device=device)
    steps = 0
    for _ in range(mask.nb_masks):
        for episode_sample in loader.sample_batches(device):
            target_q_vals, is_step_mask = target_model.forward(episode_sample)
            total_steps = is_step_mask.sum()

            mask.apply(episode_sample, loader.state_feature_names, loader.obs_feature_names)
            q_vals, _ = model.forward(episode_sample)

            loss = criterion(q_vals, target_q_vals).sum() / total_steps
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.detach()
            steps += 1

    wandb.log({
        "training/loss": running_loss.item() / steps,
        "epoch": epoch,
        "training/batch/size": loader.batch_size,
        "training/batch/in_epoch": loader.nb_batch
    }, step=epoch)
