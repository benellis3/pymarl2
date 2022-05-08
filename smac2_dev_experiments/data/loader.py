import pathlib
import torch

from components.episode_buffer import ReplayBuffer, EpisodeBatch


BATCH_DIM = 0
EPISODE_STEP_DIM = 1
AGENT_DIM = 2
FEATURE_DIM = -1


class Loader:

    def __init__(self, map_name, dataset_type, batch_size=None):
        self.dataset_dir = pathlib.Path(__file__).parent.resolve() / "masking" / map_name / dataset_type

        self.dataset_args = torch.load(self.dataset_dir / "dataset_args.pkl")
        feature_names = torch.load(self.dataset_dir / "feature_names.pkl")
        self.state_feature_names, self.obs_feature_names = feature_names["state"], feature_names["observations"]
        self.episode_buffer = torch.load(self.dataset_dir / "eval_buffer.pth")  # type: ReplayBuffer
        assert self.episode_buffer.device == "cpu"

        self.batch_size = batch_size if batch_size is not None else len(self)
        self.nb_batch = len(self) // self.batch_size

    def __len__(self):
        return self.episode_buffer.episodes_in_buffer

    def sample_batch(self, device):
        episode_sample = self.episode_buffer.sample(self.batch_size)  # type: EpisodeBatch
        max_ep_t = episode_sample.max_t_filled()
        episode_sample = episode_sample[:, :max_ep_t]
        episode_sample.to(device)
        return episode_sample


def mask_feature(features_to_match, feature_names, sample, is_state):
    if len(features_to_match) == 0:
        return
    for i, feature in enumerate(feature_names):
        for feature_to_match in features_to_match:
            if feature_to_match in feature:
                if is_state:
                    sample[:, :, i] = 0
                else:
                    sample[:, :, :, i] = 0
                break


def mask_matching(features_to_match, episode_sample,
                  state_feature_names, obs_feature_names,
                  mask_state):
    if mask_state:
        mask_feature(features_to_match, state_feature_names, episode_sample["state"], is_state=True)
    mask_feature(features_to_match, obs_feature_names, episode_sample["obs"], is_state=False)


def build_mask(mask_name, mask_state):
    features_to_mask = dict(
        nothing=[],
        everything=["_", "timestep"],
        ally_all=["ally"],
        ally_all_except_actions=[
            "ally_health", "ally_shield",
            "ally_relative", "ally_distance", "ally_visible"
            "ally_weapon", "ally_energy",
        ],
        ally_last_action_only=["ally_last_action"],
        ally_health=["ally_health"],
        ally_shield=["ally_shield"],
        ally_health_and_shield=["ally_health", "ally_shield"],
        ally_distance=["ally_distance", "ally_relative", "ally_visible"],
        enemy_all=["enemy"],
        enemy_health=["enemy_health"],
        enemy_shield=["enemy_shield"],
        enemy_health_and_shield=["enemy_health", "enemy_shield"],
        enemy_distance=["enemy_distance", "enemy_relative", "enemy_shootable"],
    )[mask_name]
    return lambda episode_sample, state_feature_names, obs_feature_names: mask_matching(features_to_mask,
                                                                                        episode_sample,
                                                                                        state_feature_names,
                                                                                        obs_feature_names,
                                                                                        mask_state)
