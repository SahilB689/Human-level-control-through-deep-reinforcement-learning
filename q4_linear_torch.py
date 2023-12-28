import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.test_env import EnvTest
from core.deep_q_learning_torch import DQN
from q3_schedule import LinearExploration, LinearSchedule

from configs.q4_linear import config
import logging


class Linear(DQN):
    def initialize_models(self):
        state_shape = self.env.state_shape()
        img_height, img_width, n_channels = state_shape
        num_actions = self.env.num_actions()

        self.q_network = nn.Linear(img_height * img_width * n_channels, num_actions)
        self.target_network = nn.Linear(img_height * img_width * n_channels, num_actions)

    def get_q_values(self, state: torch.Tensor, network: str = "q_network"):
        out = None

        if network == "q_network":
            out = self.q_network(state.view(state.size(0), -1))
        elif network == "target_network":
            out = self.target_network(state.view(state.size(0), -1))

        return out

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def calc_loss(
        self,
        q_values: torch.Tensor,
        target_q_values: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        done_mask: torch.Tensor,
    ) -> torch.Tensor:
        num_actions = self.env.num_actions()
        gamma = self.config.gamma

        Q_samp = rewards + gamma * target_q_values.max(dim=1).values * (~done_mask)
        
        actions = actions.long()
        
        Q_actual = q_values.gather(1, actions.view(-1, 1)).squeeze()

        loss = F.mse_loss(Q_actual, Q_samp)

        return loss

    def add_optimizer(self):
        self.optimizer = torch.optim.Adam(self.q_network.parameters())


if __name__ == "__main__":
    logging.getLogger(
        "matplotlib.font_manager"
    ).disabled = True  # disable font manager warnings
    env = EnvTest((5, 5, 1))

    exp_schedule = LinearExploration(
        env, config.eps_begin, config.eps_end, config.eps_nsteps
    )

    lr_schedule = LinearSchedule(config.lr_begin, config.lr_end, config.lr_nsteps)

    model = Linear(env, config)
    model.run(exp_schedule, lr_schedule, run_idx=1)
