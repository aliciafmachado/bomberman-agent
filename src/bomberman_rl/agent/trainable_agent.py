from abc import ABC

from .agent import Agent


class TrainableAgent(Agent, ABC):
    """
    Base class for an agent that takes actions on and environment
    """

    AVAILABLE_MODES = ["eval", "train"]

    def __init__(self):
        super().__init__()
        pass

    def switch_mode(self, *args):
        """
        Switches the agent mode to train
        """
        raise NotImplementedError

    def load(self, path: str):
        """
        Loads the agent trainable parameters from file
        :param path: path to file
        """
        raise NotImplementedError

    def save(self, path, name: str):
        """
        Saves the agent's trainable parameters to file
        :param name: name of the save
        :param path: path to save file
        :return:
        """
        raise NotImplementedError
