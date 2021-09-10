# CITATION: Udacity's Deep Reinforement Learning Course - DDPG Bipedal Excersize
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)


class DDPGActor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int,
        hidden_units: Tuple[int] = (256, 128, 64),
        upper_bound: float = 1,
        act_func: callable = F.relu,
    ):
        """Initialize parameters and build model.
        Parameters
        ----------
        state_size : int
            Dimension of each state
        action_size : int
            Dimension of each action
        seed : int
            Random seed
        hidden_units : tuple[int]
            Number of nodes in first, second, and third hidden layer
        upper_bound : float
            upper bound of action space to clip to
            (defines lower_bound by -x to x)

        CITATION: the alogrithm for implemeting the learn_every // update_every
                  was derived from recommendations for the continuous control
                  project as well as reviewing recommendations on the Mentor
                  help forums - Udacity's Deep Reinforement Learning Course
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.fc4 = nn.Linear(hidden_units[2], action_size)
        self.reset_parameters()

        self.upper_bound = upper_bound
        self.act_func = act_func

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.act_func(self.fc1(state))
        x = self.act_func(self.fc2(x))
        x = self.act_func(self.fc3(x))
        return torch.tanh(self.fc4(x)) * self.upper_bound


class DDPGCritic(nn.Module):
    """Critic (Value) Model."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int,
        hidden_units: tuple = (256, 128, 64),
        act_func: callable = F.relu,
    ):
        """Initialize parameters and build model.
        Params
        ======
        state_size : int
            Dimension of each state
        action_size : int
            Dimension of each action
        seed : int
            Random seed
        hidden_units : Tuple[int]
            Number of nodes in the first, second, and third hidden layers

        CITATION: the alogrithm for implemeting the learn_every // update_every
                  was derived from recommendations for the continuous control
                  project as well as reviewing recommendations on the Mentor
                  help forums - Udacity's Deep Reinforement Learning Course
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0] + action_size, hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.fc4 = nn.Linear(hidden_units[2], 1)

        self.act_func = act_func
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Build a critic (value) network that maps (state, action)
        pairs -> Q-values.
        """
        state = state.type(torch.FloatTensor)
        action = action.type(torch.FloatTensor)
        xs = self.act_func(self.fc1(state))
        x = torch.cat((xs, action), dim=1)
        x = self.act_func(self.fc2(x))
        x = self.act_func(self.fc3(x))
        return self.fc3(x)
