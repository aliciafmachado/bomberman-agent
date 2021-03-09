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

    def __init__(self, env, display_mode='print'):
        """
        :param env: The environment to be used
        """

        self._env = env
        self._display = display_mode

    def run(self):
        raise NotImplementedError
