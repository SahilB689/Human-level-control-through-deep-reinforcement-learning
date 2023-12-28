import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import get_logger
from utils.test_env import EnvTest
from q3_schedule import LinearExploration, LinearSchedule
from q4_linear_torch import Linear
import logging


from configs.q5_nature import config


class NatureQN(Linear):
    """
    Implementing DQN that will solve MinAtar's environments. 
    """

    def initialize_models(self):
        """Creates the 2 separate networks (Q network and Target network). The input
        to these models will be an img_height * img_width image
        with channels = n_channels * self.config.state_history 
        """
        state_shape = self.env.state_shape()
        img_height, img_width, n_channels = state_shape 
        num_actions = self.env.num_actions() 
        self.q_network = nn.Sequential(nn.Conv2d(n_channels, out_channels=16, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(in_features= 16 * ((img_height-3)//1 + 1) * ((img_width-3)//1 + 1), out_features=128),
        nn.ReLU(), 
        nn.Linear(in_features=128, out_features=num_actions) )
        
        self.target_network = nn.Sequential(nn.Conv2d(n_channels, out_channels=16, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),                             
        nn.Flatten(),
        nn.Linear(in_features= 16 * ((img_height-3)//1 + 1) * ((img_width-3)//1 + 1), out_features=128),
        nn.ReLU(), 
        nn.Linear(in_features=128, out_features=num_actions) )
         

    def get_q_values(self, state, network):
        """
        Returns Q values for all actions

        Args:
            state: (torch tensor)
                shape = (batch_size, img height, img width, nchannels x config.state_history)
            network: (str)
                The name of the network, either "q_network" or "target_network"

        Returns:
            out: (torch tensor) of shape = (batch_size, num_actions) 
        """
        out = None 
        
        state = state.reshape(state.shape[0], state.shape[3], state.shape[1], state.shape[2])
        
        out = torch.zeros(len(state), self.env.num_actions())
        
        if network == "q_network":  
            out = self.q_network(state)  
        elif network == "target_network":  
            out = self.target_network(state)   
                 
        return out


"""
Use deep Q network for test environment.
"""
if __name__ == "__main__":
    logging.getLogger(
        "matplotlib.font_manager"
    ).disabled = True  # disable font manager warnings
    env = EnvTest((8, 8, 6))

    # exploration strategy
    exp_schedule = LinearExploration(
        env, config.eps_begin, config.eps_end, config.eps_nsteps
    )

    # learning rate schedule
    lr_schedule = LinearSchedule(config.lr_begin, config.lr_end, config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule, run_idx=1)
