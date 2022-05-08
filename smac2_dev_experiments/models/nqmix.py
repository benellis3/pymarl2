import torch

from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from controllers.n_controller import NMAC


class NQmixNet:
    def __init__(self, loader, device, is_target=False):
        self.device = device
        self.dataset_args = loader.dataset_args
        self.is_target = is_target

        self.mac = NMAC(loader.episode_buffer.scheme, None, loader.dataset_args)
        self.mixer = Mixer(loader.dataset_args)
        self.params = list(self.mac.parameters()) + list(self.mixer.parameters())

        if self.is_target:
            self.load_models(loader.dataset_dir)

        if self.device.type == 'cuda':
            self.cuda()

    def parameters(self):
        return self.params

    def cuda(self):
        self.mac.cuda()
        self.mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        torch.save(self.mixer.state_dict(), "{}/mixer.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.mixer.load_state_dict(
            torch.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))

    def forward(self, batch: EpisodeBatch):
        if self.is_target:
            with torch.no_grad():
                return self._forward(batch)
        else:
            return self._forward(batch)

    def _forward(self, batch: EpisodeBatch):
        # Get the relevant quantities.
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t, test_mode=True)
            mac_out.append(agent_outs)
        mac_out = torch.stack(mac_out, dim=1)  # Concat over time
        chosen_action_qvals = torch.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        mask = mask.expand_as(chosen_action_qvals)
        chosen_action_qvals *= mask

        return chosen_action_qvals, mask