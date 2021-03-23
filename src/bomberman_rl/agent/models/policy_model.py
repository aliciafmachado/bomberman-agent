from bomberman_rl.envs.conventions import N_ACTIONS
import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    """
    Neural network that outputs the stochastic policy
    """
    def __init__(self, height, width, n_layers=5, n_actions=N_ACTIONS):
        # TODO do not hardcode n_layers
        """
        @param height: height of the environment frame
        @param width: width of the environment frame
        @param n_actions: Number of possible actions
        """
        super().__init__()

        # TODO add timer size

        self.__inputs = nn.Linear(height*width*n_layers, 300)
        self.__hidden1 = nn.Linear(300+10, 150)
        self.__hidden2 = nn.Linear(150+10, 60)
        self.__hidden3 = nn.Linear(60+10, 30)
        # self.__dropout = nn.Dropout(p=0.6)
        self.__out = nn.Linear(30, n_actions)

    def forward(self, x, timer):
        x = self.__inputs(x)
        x = F.relu(x)
        # x - self.__dropout(x)
        x = self.__hidden1(torch.cat((x, timer), dim=1))
        x = F.relu(x)
        x = self.__hidden2(torch.cat((x, timer), dim=1))
        x = F.relu(x)
        x = self.__hidden3(torch.cat((x, timer), dim=1))
        x = F.relu(x)
        x = self.__out(x)
        return F.softmax(x, dim=1)
