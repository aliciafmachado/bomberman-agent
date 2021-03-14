import torch
from torch.distributions import Categorical
from .trainable_agent import TrainableAgent
from .models.policy_model import Policy
from bomberman_rl.envs.conventions import PLACE_BOMB, STOP



class PolicyGradientAgent(TrainableAgent):
    def __init__(self, height, width):
        super().__init__()
        self.__policy = Policy(height, width).double()

        # Agent's resettable properties
        self.__log_probs_history = []

        self.__timer = 0

        self.__history = torch.zeros((10, height*width))

    def switch_mode(self, mode):
        super().switch_mode(mode)
        if mode == "train":
            self.__policy.train()
        elif mode == "eval":
            self.__policy.eval()

    def reset(self):
        """
        Resets the agent's parameters before each rollout
        """
        self.__log_probs_history = []
        self.__timer = 0
        self.__history = torch.zeros_like(self.__history)

    def choose_action(self, observation):
        """
        Given an observation choose an action
        :param observation: the observation given by the environment
        :return: the chosen action
        """

        if self.__timer > 0:
            self.__timer -= 1

        timer_one_hot = torch.nn.functional.one_hot(torch.tensor(self.__timer), 10)
        timer_one_hot = timer_one_hot.unsqueeze(0).double()

        flattened_obs = torch.tensor(observation).view(-1).unsqueeze(0).double()

        # Get action according to policy network
        probs = self.__policy(flattened_obs, timer_one_hot)
        dist = Categorical(probs)
        # print(dist.probs)
        action = dist.sample()

        # Saving action in history
        self.__log_probs_history.append(dist.log_prob(action))

        if action == PLACE_BOMB and self.__timer > 0:
            return STOP
            # action[0] = STOP
        if action == PLACE_BOMB:
            self.__timer = 10
            return action.item()

        return action.item()

    def get_log_probs_history(self):
        """
        :return: the history of log_probs followed by the agent
        """
        if self._mode is not "train":
            raise Exception("Can only access the history of logprobs in train mode")
        return self.__log_probs_history.copy()

    def get_policy_params(self):
        """
        :return: the params of the neural network model
        """
        if self._mode is not "train":
            raise Exception("Can only access the network params in train mode")
        return self.__policy.parameters()
