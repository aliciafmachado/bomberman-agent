import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

from bomberman_rl.simulator.base_simulator import BaseSimulator
from bomberman_rl.agent.dqn_agent import DQNAgent
from bomberman_rl.memory.replay_memory import ReplayMemory, Simulation


class DQNAgentSingleCoach(BaseSimulator):
    """
    Teaches a single DQNAgent to play alone
    """

    def __init__(self, env: gym.Env, agent: DQNAgent, n_episodes=10000, display='human',
                 batch_size=32, exploration_init=0.99, exploration_end=0.2,
                 exploration_decay=1000, max_steps=50, show_each=1000, fps=10,
                 plot_loss=False, plot_rewards=False):
        super().__init__(env, display)

        # Initialize internal variables
        assert 0 <= exploration_init <= 1 and 0 <= exploration_end <= 1
        self.__agent = agent
        self.__batch_size = batch_size
        self.__exploration_init = exploration_init
        self.__exploration_end = exploration_end
        self.__exploration_decay = exploration_decay
        self.__max_steps = max_steps
        self.__show_each = show_each
        self.__n_episodes = n_episodes
        self.__fps = fps

        self.__plot_loss = plot_loss
        self.losses = []
        self.__plot_rewards = plot_rewards
        self.rewards = []

        self.__target_update = 10
        self.__transform = transforms.ToTensor()
        self.__memory_size = 10000
        self.__memory = ReplayMemory(self.__memory_size)

    def run(self):
        """
        Perform the learning loops
        """

        # Switch agent to train mode
        self.__agent.switch_mode('train')

        for i in range(self.__n_episodes + 1):
            if not i % self.__show_each:
                self.__run_single_simulation(self._display, i)
            else:
                self.__run_single_simulation(None, i)

        # Switch agent to test mode
        self.__agent.switch_mode('eval')

    def simulate(self, n_episodes=1):
        self.__agent.switch_mode('eval')

        for i in range(n_episodes):
            self.__run_single_simulation(self._display, i)

    def __run_single_simulation(self, display, idx, verbose=False):
        """
        Does one simulation pass until the agent breaks all of the blocks or until it dies
        :param display: The kind of display to show intermediary this simulation
        """

        device = self.__agent.device
        observation = self._env.reset()
        observation = self.__transform(observation).unsqueeze(0).float().to(device)
        self.__agent.reset()
        self.__render(display, None)

        done = False
        time = torch.tensor([0], device=device)
        cum_reward = 0

        for i in range(self.__max_steps):
            # Check if it's already over
            if done:
                self.__render(display, 'End of episode')
                break

            action = self.__agent.choose_action(observation, idx,
                                                eps_decay=self.__exploration_decay,
                                                initial_eps=self.__exploration_init,
                                                end_eps=self.__exploration_end)
            next_time = self.__agent.time

            # Perform last action
            next_observation, reward, done, _ = self._env.step(action.item())
            next_observation = self.__transform(next_observation).unsqueeze(0).float().to(
                device)
            cum_reward += reward
            reward = torch.tensor([reward], device=device)

            # Store the transition in memory
            self.__memory.push(observation, action, next_observation, reward, time,
                               next_time)

            # Next state
            observation = next_observation
            time = next_time

            if len(self.__memory) >= self.__batch_size:
                sample = self.__memory.sample(self.__batch_size)
                batch = Simulation(*zip(*sample))
                loss = self.__agent.train(batch, self.__batch_size)

                if self.__plot_loss:
                    self.losses.append(loss)
                if verbose:
                    print('Episode[{}/{}], Loss: {:.4f}, Buffer state[{}/{}], Reward: {}'
                          .format(idx + 1, self.__n_episodes, loss, len(self.__memory),
                                  self.__memory_size, reward.item()))

            # Render
            self.__render(display, (idx, i, float(reward)))

        if self.__plot_rewards:
            self.rewards.append(cum_reward)

        # Update or not the targetNet
        if (idx + 1) % self.__target_update == 0:
            self.__agent.target_net.train()
            self.__agent.target_net.load_state_dict(self.__agent.q_net.state_dict())
            self.__agent.target_net.eval()

    def __render(self, display, info):
        if display is not None:
            print('\033c')
            print(info)
            self._env.render(mode=display, steps_per_sec=self.__fps)

    def plot_loss(self):
        plt.figure()
        running_mean = np.convolve(self.losses, np.ones(100) / 100, mode="valid")
        plt.plot(np.arange(len(running_mean)), running_mean)
        plt.savefig('loss.png')

    def plot_rewards(self):
        plt.figure()
        running_mean = np.convolve(self.rewards, np.ones(10) / 10, mode="valid")
        plt.plot(np.arange(len(running_mean)), running_mean)
        plt.savefig('rewards.png')
