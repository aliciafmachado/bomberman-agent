# Based on the Tutorial on DQNs from Pytorch
# Link to tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

# Importing libraries:
import torch.nn as nn
from bomberman_rl.utils.dqn_utils import conv2d_output
import torch.nn.functional as F
import torch

# The input for our neural network will be the difference between the previous
# state and the current one
class DQNModel(nn.Module):

    def __init__(self, height, width, n_dim, device, n_actions=6):
        '''
        @param height: height of the environemnt frame
        @param width: width of the environment frame
        @param n_dim: Number of frames for each state
        @param n_actions: Number of possible actions
        '''
        super().__init__()
        self.n_actions = n_actions
        self.conv1 = nn.Conv2d(n_dim, 8, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.device = device

        # We calculate the dimensions after the convolutional layers
        linear_input_size = 16 * conv2d_output(conv2d_output(conv2d_output(height))) * \
            conv2d_output(conv2d_output(conv2d_output(width)))
        
        # The linear layer that will return the output
<<<<<<< HEAD
        self.linear = nn.Linear(linear_input_size + n_actions, n_actions)
=======
        self.linear = nn.Linear(linear_input_size + 1, n_actions)
>>>>>>> Addind time variable to input for the nn

    def forward(self, x, t):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.cat([x.view(x.size(0), -1).clone().detach(), 
<<<<<<< HEAD
            F.one_hot(t, num_classes=self.n_actions).to(self.device)], dim=1)

=======
            torch.tensor(t, device=self.device)], dim=1)
        
>>>>>>> Addind time variable to input for the nn
        out = self.linear(x)
        return out