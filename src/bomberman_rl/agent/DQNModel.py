# Based on the Tutorial on DQNs from Pytorch
# Link to tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

# Importing libraries:
import torch.nn as nn
from bomberman_rl.utils.dqn_utils import conv2d_output
import torch.nn.functional as F

# The input for our neural network will be the difference between the previous
# state and the current one
class DQNModel(nn.Module):

    def __init__(self, height, width, n_dim, n_actions=6):
        '''
        @param height: height of the environemnt frame
        @param width: width of the environment frame
        @param n_dim: Number of frames for each state
        @param n_actions: Number of possible actions
        '''
        self.conv1 = nn.Conv2d(n_dim, 8, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(16)

        # We calculate the dimensions after the convolutional layers
        linear_input_size = 16 * conv2d_output(conv2d_output(height)) * \
            con2d_output(con2d_output(width))

        # The linear layer that will return the output
        self.linear = nn.Linear(linear_input_size, n_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        return self.linear(x.view(x.size(0), -1))