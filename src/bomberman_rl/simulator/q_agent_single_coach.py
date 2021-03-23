from .base_simulator import BaseSimulator
import pickle


class QAgentSingleCoach(BaseSimulator):
    """
    Teaches a single QAgent to play alone
    """

    def __init__(self, env, agent,
                 display="none",
                 lr=0.2,
                 gamma=0.95,
                 exploration_factor=0.3,
                 max_steps=int(1e3),
                 show_each=10000,
                 nb_passes=100000,
                 fps=10):
        """
        :param env: The environment to be used
        :param agent: The agent to be trained
        :param display: if you wish to show the result of the training
        :param lr: learning rate
        :param gamma: factor importance of future rewards
        :param exploration_factor: controls how much the agent should explore the environment
        :param max_steps: the maximum number of steps a single simulation can have
        :param show_each: show the result of teh training each show_each passes
        :param nb_passes: the total number of game simulations to be run
        :param fps: if display is not none the fps of teh simulation to be shown
        """
        super().__init__(env, display)

        # Initialize internal variables
        assert 0 <= lr <= 1 and 0 <= gamma <= 1 and 0 <= exploration_factor <= 1
        self.__agent = agent
        self.__lr = lr
        self.__gamma = gamma
        self.__exploration_factor = exploration_factor
        self.__max_steps = max_steps
        self.__show_each = show_each
        self.__nb_passes = nb_passes
        self.__fps = fps

    def run(self):
        """
        Perform the learning loops
        """

        # Switch agent to train mode
        self.__agent.switch_mode("train")

        rewards = []

        for i in range(self.__nb_passes):
            if not i % self.__show_each:
                rewards.append(self.__run_single_simulation(self._display))
            else:
                rewards.append(self.__run_single_simulation("none"))

        # Saving reward history
        with open("rewards_q_agent.pickle", 'wb') as handle:
            pickle.dump(rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Switch agent to test mode
        self.__agent.switch_mode("eval")

    def __run_single_simulation(self, display):
        """
        Does one simulation pass until the agent breaks all of the blocks or until it dies
        :param display: The kind of display to show intermediary this simulation
        :return: reward for simulation
        """

        observation = self._env.reset()
        self.__agent.reset()
        self.__render(display, None)

        # Getting first action
        self.__action = self.__agent.choose_action(observation, self.__exploration_factor)
        self.__done = False

        # Record reward for epoch
        cumulative_reward = 0

        for i in range(self.__max_steps):
            # Check if it's already over
            if self.__done:
                self.__render(display, "End of game")
                # print(self.__agent.get_q_table_size())
                return cumulative_reward

            # Perform last action
            observation, reward, self.__done, _ = self._env.step(self.__action)

            cumulative_reward += reward

            # Get new action
            self.__action = self.__agent.choose_action(observation, self.__exploration_factor)

            # Learn from experience
            self.__agent.update_q_table(reward, self.__lr, self.__gamma)

            # Render
            self.__render(display, (i, reward, self.__agent.get_q_table_size()))

        self.__render(display, "End of game")

        return cumulative_reward

    def __render(self, display, info):
        # TODO mode this method to renderer
        if display is not "none":
            print('\033c')
            print(info)
            self._env.render(mode=display, steps_per_sec=self.__fps)
