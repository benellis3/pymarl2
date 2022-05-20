import pathlib
import torch
import sys

from components.episode_buffer import ReplayBuffer, EpisodeBatch

BATCH_DIM = 0
EPISODE_STEP_DIM = 1
AGENT_DIM = 2
FEATURE_DIM = -1

EPISODE_TO_OBS_SIZE_RATIO = 3.2  # Safe margin. Magic number.


class Loader:

    def __init__(self, map_name, dataset_type, batch_size=None, smac_version=1):
        self.map_name = map_name
        self.dataset_dir = pathlib.Path(__file__).parent.resolve() / f"smac{smac_version}" / map_name / dataset_type
        self.dataset_args = torch.load(self.dataset_dir / "dataset_args.pkl")

        feature_names = torch.load(self.dataset_dir / "feature_names.pkl")
        self.state_feature_names, self.obs_feature_names = feature_names["state"], feature_names["observations"]
        self.episode_buffer = torch.load(self.dataset_dir / "eval_buffer.pth")  # type: ReplayBuffer
        self.episode_buffer.pin_memory()
        self.max_batch_size = self.get_max_batch_size()
        self.batch_size = self.closest_to_batch_size(batch_size) if batch_size else self.guess_largest_batch_size()
        self.nb_batch = len(self) // self.batch_size
        print(f"{dataset_type} dataset. Max batch_size {self.max_batch_size}. Effective batch size {self.batch_size}.")

    def __len__(self):
        return self.episode_buffer.episodes_in_buffer

    def closest_to_batch_size(self, batch_size):
        max_size = self.max_batch_size / 2  # Training consumes memory for autograd and so on ...
        if batch_size <= max_size:
            return batch_size
        else:
            while batch_size > max_size:
                batch_size //= 2
            return batch_size

    def get_max_batch_size(self):
        device_memory = torch.cuda.get_device_properties(0).total_memory
        obs = self.episode_buffer["obs"]
        size_of_one_obs = sys.getsizeof(obs.storage()) / len(self)
        size_of_one_episode = size_of_one_obs * EPISODE_TO_OBS_SIZE_RATIO
        max_batch_size = device_memory // size_of_one_episode
        return max_batch_size

    def guess_largest_batch_size(self):
        if self.max_batch_size >= len(self):
            return len(self)
        else:
            possible_batch_sizes = [4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
            batch_size = max(filter(lambda size: size <= self.max_batch_size, possible_batch_sizes))
            return batch_size

    def sample_batches(self, device):
        for _ in range(self.nb_batch):
            episode_sample = self.episode_buffer.sample(self.batch_size)  # type: EpisodeBatch
            episode_sample.to(device)
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]
            yield episode_sample

    def get_validation_batches(self, device, max_seq_length=None):
        for i in range(self.nb_batch):
            episode_sample = self.episode_buffer[i*self.batch_size: (i+1)*self.batch_size]
            episode_sample.to(device)
            if max_seq_length is not None:
                episode_sample = episode_sample[:, :max_seq_length]
            yield episode_sample


class Mask:
    def __init__(self, obs_features=None, state_features="same_as_obs", exception=None):
        if obs_features is None:
            self.obs_features = []
        else:
            self.obs_features = obs_features
        if state_features == "same_as_obs":
            self.state_features = self.obs_features
        else:
            self.state_features = state_features
        self.exception = exception
        self.nb_masks = 1

    @staticmethod
    def mask_feature(mask_substrings, exception, feature_names, sample, is_state):
        if len(mask_substrings) == 0:
            return
        for i, feature in enumerate(feature_names):
            for mask_substring in mask_substrings:
                if mask_substring in feature:
                    if exception is None or exception not in feature:
                        if is_state:
                            sample[:, :, i] = 0
                        else:
                            sample[:, :, :, i] = 0
                        break

    @staticmethod
    def mask_(feature_name, mask_substrings, sample, exception, is_state):
        i, feature = feature_name
        for mask_substring in mask_substrings:
            if mask_substring in feature:
                if exception is None or exception not in feature:
                    if is_state:
                        sample[:, :, i] = 0
                    else:
                        sample[:, :, :, i] = 0
                    return

    @staticmethod
    def mask_feature_v2(mask_substrings, exception, feature_names, sample, is_state):
        if len(mask_substrings) == 0:
            return

        map(lambda feature_name: Mask.mask_(feature_name, mask_substrings, sample, exception, is_state), feature_names)

    def apply(self, episode_sample, state_feature_names, obs_feature_names):
        self.mask_feature(self.state_features, self.exception, state_feature_names, episode_sample["state"],
                          is_state=True)
        self.mask_feature(self.obs_features, self.exception, obs_feature_names, episode_sample["obs"],
                          is_state=False)


features_to_mask_map = dict(
    nothing=Mask(),
    everything=Mask(["_", "timestep"]),
    ally_all=Mask(["ally"]),
    ally_all_except_actions=Mask(["ally"], exception="action"),
    ally_last_action_only=Mask(["ally_last_action"]),
    ally_health=Mask(["ally_health"]),
    ally_shield=Mask(["ally_shield"]),
    ally_health_and_shield=Mask(["ally_health", "ally_shield"]),
    ally_distance=Mask(["ally_distance", "ally_relative", "ally_visible"]),
    enemy_all=Mask(["enemy"]),
    enemy_health=Mask(["enemy_health"]),
    enemy_shield=Mask(["enemy_shield"]),
    enemy_health_and_shield=Mask(["enemy_health", "enemy_shield"]),
    enemy_distance=Mask(["enemy_distance", "enemy_relative", "enemy_shootable"]),
    own_health=Mask(["own_health"], state_features=["ally_heath"])
)


class RandomMask:
    def __init__(self):
        self.masks = list(features_to_mask_map.values())
        self.nb_masks = len(self.masks)

    def apply(self, episode_sample, state_feature_names, obs_feature_names):
        mask_index = torch.randint(0, len(self.masks), (1,)).item()
        self.masks[mask_index].apply(episode_sample, state_feature_names, obs_feature_names)


def build_mask(mask_name):
    if mask_name == "random":
        return RandomMask()
    else:
        return features_to_mask_map[mask_name]
