from bomberman_rl.envs.conventions import N_ACTIONS
import torch
import torch.nn as nn
import torch.nn.functional as F
from bomberman_rl.utils.dqn_utils import conv2d_output


class ActorCritic(nn.Module):
    """
    Neural network that outputs the stochastic policy
    """
    def __init__(self, height, width, n_dim, device, time_size, n_actions=6):
        # TODO do not hardcode n_layers
        """
        @param height: height of the environment frame
        @param width: width of the environment frame
        @param n_actions: Number of possible actions
        """
        super().__init__()

        # TODO add timer size
        self.n_actions = n_actions
        self.conv1 = nn.Conv2d(n_dim, 16, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=True)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=True)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=True)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv11 = nn.Conv2d(n_dim, 16, kernel_size=3, stride=1)
        self.bn11 = nn.BatchNorm2d(16)
        self.conv21 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=True)
        self.bn21 = nn.BatchNorm2d(32)
        self.conv31 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=True)
        self.bn31 = nn.BatchNorm2d(64)
        self.conv41 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=True)
        self.bn41 = nn.BatchNorm2d(64)
        self.device = device


        # self.__dropout = nn.Dropout(p=0.6)

        # We calculate the dimensions after the convolutional layers
        linear_input_size = 64 * conv2d_output(height) * conv2d_output(width)

        # The linear layer that will return the output
        self.time_size = time_size
        self.linear = nn.Linear(linear_input_size + time_size, 200)
        self.linear2 = nn.Linear(200, 50)
        self.linear1 = nn.Linear(linear_input_size + time_size, 200)
        self.linear21 = nn.Linear(200, 50)

        # Actor's layer
        self.__action_head = nn.Linear(50, n_actions)

        # Critic's layer
        self.__value_head = nn.Linear(50, 1)

        self.to(device)

    def reset(self):
        pass
        # self.__memory = []
        # self.__reset = True

    def forward(self, inp, timer):
        x1 = F.relu(self.bn1(self.conv1(inp)))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1 = F.relu(self.bn3(self.conv3(x1)))
        x1 = F.relu(self.bn4(self.conv4(x1)))
        x1 = torch.cat([x1.view(x1.size(0), -1).clone().detach(),
                       F.one_hot(timer, num_classes=self.time_size).to(self.device)], dim=1)

        out = F.relu(self.linear(x1))
        out = F.relu(self.linear2(out))

        action_probs = F.softmax(self.__action_head(out), dim=-1)

        x1 = F.relu(self.bn1(self.conv1(inp)))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1 = F.relu(self.bn3(self.conv3(x1)))
        x1 = F.relu(self.bn4(self.conv4(x1)))
        x1 = torch.cat([x1.view(x1.size(0), -1).clone().detach(),
                        F.one_hot(timer, num_classes=self.time_size).to(self.device)], dim=1)

        out = F.relu(self.linear(x1))
        out = F.relu(self.linear2(out))

        action_probs = F.softmax(self.__action_head(out), dim=-1)

        x2 = F.relu(self.bn1(self.conv1(inp)))
        x2 = F.relu(self.bn2(self.conv2(x2)))
        x2 = F.relu(self.bn3(self.conv3(x2)))
        x2 = F.relu(self.bn4(self.conv4(x2)))
        x2 = torch.cat([x2.view(x2.size(0), -1).clone().detach(),
                        F.one_hot(timer, num_classes=self.time_size).to(self.device)], dim=1)

        out1 = F.relu(self.linear(x2))
        out1 = F.relu(self.linear2(out1))

        # print(action_probs)
        state_value = self.__value_head(out1)

        return action_probs, state_value
