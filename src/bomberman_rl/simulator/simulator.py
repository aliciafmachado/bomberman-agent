import gym
import time

from bomberman_rl.envs.conventions import UP, LEFT, RIGHT, DOWN, STOP, PLACE_BOMB


class Simulator:
    """
    Simulates given actions
    """

    actions_dict = {
        STOP: "STOP",
        UP: "UP",
        DOWN: "DOWN",
        LEFT: "LEFT",
        RIGHT: "RIGHT",
        PLACE_BOMB: "PLACE_BOMB"
    }

    def __init__(self, env_name, actions, speed):
        """
        :param env_name: The environment to be used
        :param actions: The sequence of actions (UP, DOWN, LAFT, RIGHT, STOP or PLACE_BOMB)
        :param speed: The speed factor (1x, 2x ...) 1x = 1fps
        """
        self.__actions = actions
        self.__speed = speed
        self.__env = gym.make(env_name, display='print')

    def run(self):
        print('\033c')
        for i in range(len(self.__actions)):
            print("Action: ", Simulator.actions_dict[self.__actions[i]])
            print("Step: ", i)
            _, reward, done, _ = self.__env.step(self.__actions[i])
            print("Reward", reward)
            print("Done", done)
            self.__env.render()
            time.sleep(1/self.__speed)
            print('\033c')

    def reset(self, actions):
        self.__actions = actions
        self.__env.reset()
