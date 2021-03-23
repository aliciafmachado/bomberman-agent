import torch.nn as nn
import torch.nn.functional as F
import torch

from bomberman_rl.utils.dqn_utils import conv2d_output


class DQNModel(nn.Module):
    """
    The input for our neural network will be the difference between the previous state and
    the current one.

    Based on the Tutorial on DQNs from Pytorch
    Link to tutorial:
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """

    def __init__(self, observation_shape, n_actions, device, time_size,
                    temporal_mode):
        """
        @param observation_shape: shape of the environment frame
        @param n_actions: Number of possible actions
        @param device: cuda or cpu device
        """
        super().__init__()
        self.n_actions = n_actions
        self.temporal_mode = temporal_mode
        self.conv1 = nn.Conv2d(observation_shape[2], 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=True)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=True)
        self.device = device

        self.conv_r1 = nn.Sequential(
            nn.Conv2d(observation_shape[2], 32, 1, stride=2),
            nn.BatchNorm2d(32),
        )

        self.conv_r2 = nn.Sequential(
            nn.Conv2d(32, 64, 1, stride=2),
            nn.BatchNorm2d(64),
        )

        # We calculate the dimensions after the convolutional layers
        linear_input_size = 64 * conv2d_output(
            conv2d_output(observation_shape[0], 1, 2), 1, 2) * conv2d_output(
            conv2d_output(observation_shape[1], 1, 2), 1, 2)

        # The linear layer that will return the output
        self.time_size = time_size

        if temporal_mode == "time":
            self.linear = nn.Linear(linear_input_size + self.time_size, 200)
        else:  
            self.linear = nn.Linear(linear_input_size, 200)
        
        self.linear2 = nn.Linear(200, 50)
        self.linear3 = nn.Linear(50, n_actions)

        self.to(device)

    def forward(self, x, t):
        residual = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        residual = self.conv_r1(residual)
        x += residual
        residual = x
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        residual = self.conv_r2(residual)
        x += residual

        if self.temporal_mode == "time":
            x = torch.cat([x.view(x.size(0), -1).clone().detach(),
                    F.one_hot(t, num_classes=self.time_size).to(self.device)], dim=1)
        else:
            x = x.view(x.size(0), -1).clone().detach()

        out = F.relu(self.linear(x))
        out = F.relu(self.linear2(out))
        return self.linear3(out)
