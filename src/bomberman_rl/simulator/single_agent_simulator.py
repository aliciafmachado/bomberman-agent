import numpy as np
import time
from .base_simulator import BaseSimulator

from bomberman_rl.envs.conventions import BLOCK


class SingleAgentSimulator(BaseSimulator):
    """
    Plays a single agent on the screen
    """
    def __init__(self, env, agent,
                 display="stdout",
                 nb_runs=1,
                 max_steps=int(1e3),
                 fps=1):
        """
        :param env: The environment to be used
        :param agent: The agent to be trained
        :param display: if you wish to show the result of the training
        :param nb_runs: the total number of game simulations to be run
        :param max_steps: the maximum number of steps a run simulation can take
        :param fps: if display is not none the fps of teh simulation to be shown
        """
        super().__init__(env, display)

        # Initialize internal variables
        self.__agent = agent
        self.__nb_runs = nb_runs
        self.__max_steps = max_steps
        self.__fps = fps

    def run(self):
        """
        Perform the simulations
        """

        # Switch agent to test mode
        self.__agent.switch_mode("eval")

        for i in range(self.__nb_runs):
            self.__run_single_simulation(self._display)

    def __run_single_simulation(self, display):
        """
        Does one simulation pass until the agent breaks all of the blocks or until it dies
        :param display: The kind of display to show intermediary this simulation
        """

        observation = self._env.reset()
        self.__agent.reset()
        self.__render(display, None)

        # Getting first action
        self.__action = self.__agent.choose_action(observation)
        self.__done = False

        for i in range(self.__max_steps):
            # Check if it's already over
            if not np.any(observation[:, :, BLOCK]) or self.__done:
                return

            # Perform last action
            observation, reward, self.__done, _ = self._env.step(self.__action)

            # Get new action
            self.__action = self.__agent.choose_action(observation)

            # Render
            self.__render(display, (i, reward))

        self.__render(display, "End")

    def __render(self, display, info):
        if display is not "none":
            print(info)
            self._env.render(mode=display, steps_per_sec=self.__fps)
