# Based on the Tutorial on DQNs from Pytorch
# Link to tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

# Importing libraries:
import torch.nn as nn
from bomberman-agent.utils.dqn_utils import conv2d_output


class DQNModel(nn.Module):

    def __init__(self, height, width, n_actions, n_dim):
        self.conv1 = nn.Conv2d(n_dim, 8, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(16)

        # TODO: calculate linear dimensions
        self.linear = nn.Linear()

    def forward():
