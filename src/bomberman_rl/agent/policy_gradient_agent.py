import torch
from torch.distributions import Categorical
from .trainable_agent import TrainableAgent
from .models.policy_model import Policy


class PolicyGradientAgent(TrainableAgent):
    def __init__(self, height, width):
        super().__init__()
        self.__policy = Policy(height, width).double()

        # Agent's resettable properties
        self.__log_probs_history = []

    def switch_mode(self, *args):
        pass

    def reset(self):
        """
        Resets the agent's parameters before each rollout
        """
        self.__log_probs_history = []

    def choose_action(self, observation):
        """
        Given an observation choose an action
        :param observation: the observation given by the environment
        :return: the chosen action
        """

        # Convert the observation to pytorch
        obs = torch.tensor(observation).view(-1).double()

        # Get action according to policy network
        probs = self.__policy(obs)
        dist = Categorical(probs)
        action = dist.sample()

        # Saving action in history
        self.__log_probs_history.append(dist.log_prob(action))

        return action.item()
