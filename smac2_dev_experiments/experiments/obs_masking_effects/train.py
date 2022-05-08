import wandb


def train_single_transform(epoch, model, target_model, loader, mask_transform, criterion, optimizer, device):
    for batch_idx in range(loader.nb_batch):
        episode_sample = loader.sample_batch(device)
        target_q_vals, is_step_mask = target_model.forward(episode_sample)
        total_steps = is_step_mask.sum()

        mask_transform(episode_sample, loader.state_feature_names, loader.obs_feature_names)
        q_vals, _ = model.forward(episode_sample)

        loss = criterion(q_vals, target_q_vals).sum() / total_steps
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    wandb.log({
        "training/loss": loss.item(),
        "epoch": epoch,
        "training/batch/size": loader.batch_size,
        "training/batch/in_epoch": loader.nb_batch
    }, step=epoch)
