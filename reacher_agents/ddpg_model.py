# CITATION: Udacity's Deep Reinforement Learning Course - DDPG Bipedal Excersize
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

    def __init__(self, state_size, action_size, seed, fc_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (int): Number of nodes in first and second hidden layer

        CITATION: the alogrithm for implemeting the learn_every // update_every 
                  was derived from recommendations for the continuous control
                  project as well as reviewing recommendations on the Mentor
                  help forums - Udacity's Deep Reinforement Learning Course
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.bn = nn.BatchNorm1d(fc_units)
        self.fc1 = nn.Linear(state_size, fc_units)
        self.fc2 = nn.Linear(fc_units, fc_units)
        self.fc3 = nn.Linear(fc_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.leaky_relu(self.fc1(state))
        x = self.bn(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.bn(x)
        return F.tanh(self.fc3(x))


class DDPGCritic(nn.Module):
    """Critic (Value) Model."""

    def __init__(
        self,
        state_size,
        action_size,
        seed,
        fc_units=256,
    ):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer

        CITATION: the alogrithm for implemeting the learn_every // update_every 
                  was derived from recommendations for the continuous control
                  project as well as reviewing recommendations on the Mentor
                  help forums - Udacity's Deep Reinforement Learning Course
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.bn = nn.BatchNorm1d(fc_units)
        self.fc1 = nn.Linear(state_size + action_size, fc_units)
        self.fc2 = nn.Linear(fc_units, fc_units)
        self.fc3 = nn.Linear(fc_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Build a critic (value) network that maps (state, action)
        pairs -> Q-values.
        """
        # x = torch.cat((x, action), dim=1)
        x = torch.cat((state, action), dim=1)
        x = F.leaky_relu(self.fc1(x))
        x = self.bn(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.bn(x)
        return self.fc3(x)
