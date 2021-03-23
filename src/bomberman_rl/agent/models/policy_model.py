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

        self.__inputs = nn.Linear(height*width*n_layers, 400)
        self.__hidden1 = nn.Linear(400+10, 300)
        self.__hidden2 = nn.Linear(300, 150)
        self.__hidden3 = nn.Linear(60, 30)  
        # self.__dropout = nn.Dropout(p=0.6)

        # Actor's layer
        self.__action_head = nn.Linear(30, n_actions)

        # Critic's layer
        self.__value_head = nn.Linear(30, 1)

        self.__lstm = nn.LSTM(160, 60)

        # Reset
        self.__reset = True

        self.__memory = []

    def reset(self):
        self.__memory = []
        self.__reset = True

    def forward(self, x, timer):
        x = self.__inputs(x)
        x = F.relu(x)
        # x - self.__dropout(x)
        x = self.__hidden1(torch.cat((x, timer), dim=1))
        x = F.relu(x)
        x = self.__hidden2(x.view(1, -1))
        x = F.relu(x)
        # Checking if in reset mode
        self.__memory.append(torch.cat((x, timer), dim=1))
        x = torch.stack(self.__memory[-10:])
        out, (x, _) = self.__lstm(x)
        x = self.__hidden3(x)
        x = F.relu(x)

        action_probs = F.softmax(self.__action_head(x), dim=-1)
        state_value = self.__value_head(x)

        return action_probs, state_value
