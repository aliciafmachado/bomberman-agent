import numpy as np
from .trainable_agent import TrainableAgent
from bomberman_rl.envs.conventions import PLACE_BOMB


class QAgent(TrainableAgent):
    """
    Classical Q-Learning agent
    """

    def __init__(self):
        super().__init__()
        self.__q_table = {}
        self.__mode = "train"
        self.__last_state_action = None
        self.__curr_state_action = None
        self.__last_max_value = None
        self.__last_reward = None
        self.__timer = 0
        self.__max_timer = 6
        self.__initial_reward = 0

    def choose_action(self, observation, exploration_factor=None):
        """
        Chooses an action for the agent
        :param observation: fromm the environment
        :param exploration_factor: if in train mode, controls the probability of choosing a random action
        :return: the choice of action
        """

        if self.__mode == 'train' and exploration_factor is None:
            raise ValueError(
                'Agent must receive exploration factor to choose action in train mode')

        chosen_action = None
        best_action = None

        possible_actions = QAgent.AVAILABLE_ACTIONS.copy()
        if self.__timer > 0:
            possible_actions.remove(PLACE_BOMB)

        # Getting the best possible action according to the Q-table
        max_value = -float("inf")
        for action in possible_actions:
            state_action = (tuple(observation.reshape(-1)), self.__timer, action)
            value = self.__initial_reward
            if state_action in self.__q_table:
                value = self.__q_table[state_action]
            else:
                self.__q_table[state_action] = self.__initial_reward

            if value > max_value:
                best_action = action
                max_value = value

        if self.__mode == "train":
            # If in train mode we can choose a random action as well
            if np.random.choice(2, p=[exploration_factor, 1 - exploration_factor]):
                chosen_action = best_action
            else:
                chosen_action = np.random.choice(possible_actions)

            # Update internal memory
            self.__last_state_action = self.__curr_state_action
            self.__curr_state_action = (tuple(observation.reshape(-1)), self.__timer, chosen_action)
            self.__last_max_value = max_value
        elif self.__mode == "eval":
            chosen_action = best_action

        # Updating bomb timers
        if chosen_action == PLACE_BOMB:
            self.__timer = self.__max_timer
        elif self.__timer > 0:
            self.__timer -= 1

        return chosen_action

    def update_q_table(self, last_reward, lr, gamma):
        """
        Given the reward for the previous cycle update the state action value
        of the previous cycle in the q_table
        :param last_reward: the reward given for this agent in the last cycle
        :param lr: learning rate
        :param gamma: discount factor
        :return: update succeeded
        """

        # The model should be in train mode
        if self.__mode != "train":
            raise Exception("The model should be in train mode to update the Q-table")

        # Update the table only if there was a last state action, this
        # means that at least 2 actions should have been taken
        success = False
        if self.__last_state_action:
            success = True
            # Initializing Q-table if needed
            if self.__last_state_action not in self.__q_table:
                self.__q_table[self.__last_state_action] = self.__initial_reward

            # Updating Q-table
            self.__q_table[self.__last_state_action] += lr * (last_reward + gamma * self.__last_max_value
                                                              - self.__q_table[self.__last_state_action])

        return success

    def reset(self):
        """
        Resets parameters for new run
        """
        self.__curr_state_action = None
        self.__last_max_value = None
        self.__last_state_action = None
        self.__last_reward = None
        self.__timer = 0

    def switch_mode(self, mode):
        if mode not in QAgent.AVAILABLE_MODES:
            raise ValueError
        self.__mode = mode

    def get_q_table_size(self):
        return len(self.__q_table)
