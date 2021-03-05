import time
from .base_simulator import BaseSimulator


class RunActionsSimulator(BaseSimulator):
    """
    Simulates given actions
    """

    def __init__(self, env_name, actions, fps):
        """
        :param env_name: The environment to be used
        :param actions: The sequence of actions (UP, DOWN, LAFT, RIGHT, STOP or PLACE_BOMB)
        :param fps: the fps of the simulation
        """
        super().__init__(env_name)
        self.__actions = actions
        self.__fps = fps

    def run(self):
        print('\033c')
        for i in range(len(self.__actions)):
            print("Action: ", BaseSimulator.ACTIONS_DICT[self.__actions[i]])
            print("Step: ", i)
            _, reward, done, _ = self._env.step(self.__actions[i])
            print("Reward", reward)
            print("Done", done)
            self._env.render()
            time.sleep(1 / self.__fps)
            print('\033c')

    def reset(self, actions):
        self.__actions = actions
        self._env.reset()
