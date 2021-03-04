import numpy as np
from .agent import Agent


class RandomAgent(Agent):
    """
    This agent takes always a random action
    """

    def __init__(self, distribution=np.array([1 / len(Agent.AVAILABLE_ACTIONS)] * len(Agent.AVAILABLE_ACTIONS))):
        super().__init__()
        """
        The random agent should receive a probability distribution for its actions,
        it'll assume an uniform distribution otherwise
        """
        assert len(distribution) == len(Agent.AVAILABLE_ACTIONS)
        self.__distribution = distribution

    def choose_action(self, *args):
        """
        Can take any parameters, ignore them all and take a random action
        """
        return np.random.choice(Agent.AVAILABLE_ACTIONS, p=self.__distribution)
