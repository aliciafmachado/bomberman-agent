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

    def __init__(self, height, width, n_dim, device, time_size, n_actions=6):
        '''
        @param height: height of the environemnt frame
        @param width: width of the environment frame
        @param n_dim: Number of frames for each state
        @param n_actions: Number of possible actions
        '''
        super().__init__()
        self.n_actions = n_actions
        self.conv1 = nn.Conv2d(n_dim, 16, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=True)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=True)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=True)
        self.bn4 = nn.BatchNorm2d(64)
        self.device = device
        self.to(device)

        # We calculate the dimensions after the convolutional layers
        linear_input_size = 64 * conv2d_output(height) * conv2d_output(width)
        
        # The linear layer that will return the output
        self.time_size = time_size
        self.linear = nn.Linear(linear_input_size + time_size, 200)
        self.linear2 = nn.Linear(200, 50)
        self.linear3 = nn.Linear(50, n_actions)

    def forward(self, x, t):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = torch.cat([x.view(x.size(0), -1).clone().detach(), 
            F.one_hot(t, num_classes=self.time_size).to(self.device)], dim=1)

        out = F.relu(self.linear(x))
        out = F.relu(self.linear2(out))
        return self.linear3(out)