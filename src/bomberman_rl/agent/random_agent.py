import numpy as np
from .agent import Agent

from ..envs.conventions import UP, DOWN, LEFT, RIGHT, PLACE_BOMB


class RandomAgent(Agent):
    """
    This agent takes always a random action
    """

    available_actions = [UP, DOWN, LEFT, RIGHT, PLACE_BOMB]

    def __init__(self, distribution=np.array([1 / len(available_actions)] * len(available_actions))):
        super().__init__()
        assert len(distribution) == len(RandomAgent.available_actions)
        self.__distribution = distribution
        self.__q_table = {}
        self.__mode = "train"

    def choose_action(self):
        return np.random.choice(RandomAgent.available_actions, p=self.__distribution)
