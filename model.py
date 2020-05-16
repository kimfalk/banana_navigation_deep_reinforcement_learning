import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """Deep Q-Network"""

    def __init__(self, state_size: int, action_size: int, seed, fc1_units=64, fc2_units=64):
        """
        Instantiate parameters and construct model.

        :param state_size:  Dimension of a state
        :param action_size: Dimension of an action
        :param seed:        Random seed (to enable reproduction of results)
        :param fc1_units:   Size of first hidden layer
        :param fc2_units:   Size of second hidden layer
        """

        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
