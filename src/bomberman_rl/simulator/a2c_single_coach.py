import numpy as np
import pickle
import torch
import torch.optim as optim
import torch.nn.functional as F
from bomberman_rl.envs.conventions import BLOCK
from .base_simulator import BaseSimulator


class A2CSingleCoach(BaseSimulator):
    """
    Teaches a single A2C to play alone
    """

    def __init__(self, env_name, agent,
                 display="none",
                 gamma=0.95,
                 max_steps_per_pass=100,
                 show_each=1000,
                 nb_passes=10000,
                 fps=10):
        """
        :param env_name: The name of the environment to be used
        :param agent: The agent to be trained
        :param display: if you wish to show the result of the training
        :param gamma: discount factor for future rewards
        :param max_steps_per_pass: the maximum number of steps a single simulation can have
        :param show_each: show the result of teh training each show_each passes
        :param nb_passes: the total number of game simulations to be run
        :param fps: if display is not none the fps of teh simulation to be shown
        """
        super().__init__(env_name, display)

        # Initialize internal variables
        assert 0 <= gamma <= 1
        self.__agent = agent
        self.__gamma = gamma
        self.__max_steps = max_steps_per_pass
        self.__show_each = show_each
        self.__nb_passes = nb_passes
        self.__fps = fps
        self.__eps = np.finfo(np.float32).eps.item()

        # Setting up optimizer
        # TODO get it's parameters in constructor
        self.__optimizer = None

    def run(self):
        """
        Perform the learning loops
        """

        # Switch agent to train mode
        self.__agent.switch_mode("train")

        # Setting up optimizer
        params = self.__agent.get_policy_params()
        self.__optimizer = optim.Adam(params, lr=1e-4)

        rewards = []

        # Rolling out passes
        for i in range(self.__nb_passes):
            if not i % self.__show_each:
                rewards.append(self.__run_single_simulation(self._display))
            else:
                rewards.append(self.__run_single_simulation("none"))

        # Saving reward history
        with open("rewards_a2c_agent.pickle", 'wb') as handle:
            pickle.dump(rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Switch agent back to evaluation mode
        self.__agent.switch_mode("eval")

    def __run_single_simulation(self, display):
        """
        Does one simulation pass until the agent breaks all of the blocks or until it dies
        :param display: The kind of display to show intermediary this simulation
        """

        observation = self._env.reset()
        self.__agent.reset()
        pass_rewards = []

        # Performing pass until the end of the game
        for t in range(self.__max_steps):
            # Get agent's action
            action = self.__agent.choose_action(observation)

            # Getting environment's response
            observation, reward, done, _ = self._env.step(action)

            # Recording the rewards for this pass
            pass_rewards.append(reward)

            # Render
            self.__render(display, (t, reward))

            # Check if it's already over
            if not np.any(observation[:, :, BLOCK]) or done:
                break

        pass_sum = sum(pass_rewards)

        # Calculating the discounted rewards
        R = 0
        returns = []
        for r in pass_rewards[::-1]:
            R = r + self.__gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.__eps)

        # Calculate the losses for current rollout
        policy_losses = []
        value_losses = []
        log_probs_history, values_history = self.__agent.get_log_actions_history()
        for log_prob, value, R in zip(log_probs_history, values_history, returns):
            advantage = R - value.item()

            # Calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # Calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value.view(-1), torch.tensor([R])))

        # Reset gradients
        self.__optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        # Perform backprop
        loss.backward()
        self.__optimizer.step()
        # if display == "none":
        #     print(loss)

        return pass_sum

    def __render(self, display, info):
        # TODO mode this method to renderer
        if display is not "none":
            # print('\033c')
            print(info)
            self._env.render(mode=display, steps_per_sec=self.__fps)
