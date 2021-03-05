import pickle
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

    @classmethod
    def load(cls, path, name):
        """
        Loads the an agent from a file
        :param path: path to file
        :param name: agent's name
        """
        with open(path+name+".pickle", 'rb') as handle:
            return pickle.load(handle)

    def save(self, path, name):
        """
        Saves the current agent to file
        :param name: name of the save
        :param path: path to save file
        :return:
        """
        path = path + name + ".pickle"
        with open(path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
