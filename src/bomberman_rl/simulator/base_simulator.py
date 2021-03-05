import gym

from bomberman_rl.envs.conventions import UP, LEFT, RIGHT, DOWN, STOP, PLACE_BOMB


class BaseSimulator:
    """
    Runs the game
    """

    ACTIONS_DICT = {
        STOP: "STOP",
        UP: "UP",
        DOWN: "DOWN",
        LEFT: "LEFT",
        RIGHT: "RIGHT",
        PLACE_BOMB: "PLACE_BOMB"
    }

    def __init__(self, env_name, display_mode='print'):
        """
        :param env_name: The environment to be used
        """

        self._env = gym.make(env_name)
        self._display = display_mode

    def run(self):
        raise NotImplementedError
