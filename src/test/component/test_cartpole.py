from collections import namedtuple
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch import optim

from bomberman_rl.agent.trainable_agent import TrainableAgent
from bomberman_rl.simulator.base_simulator import BaseSimulator


CartpoleSimulation = namedtuple('Simulation', ('state', 'action', 'next_state', 'reward'))


class CartpoleReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a simulation."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = CartpoleSimulation(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class CartpoleDQNModel(nn.Module):
    def __init__(self, device, n_actions=2):
        super().__init__()
        self.n_actions = n_actions
        self.device = device

        hidden_dim = 50
        self.linear = nn.Linear(4, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, n_actions)

        self.to(device)

    def forward(self, x):
        x = F.relu(self.linear(x))
        return self.linear2(x)


class CartpoleDQNAgent(TrainableAgent):
    def __init__(self, n_actions=2, lr=1e-3, gamma=0.9):
        super().__init__()

        # Temporary variable
        self.mode = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_net = CartpoleDQNModel(self.device, n_actions)
        self.q_net = CartpoleDQNModel(self.device, n_actions)
        self.n_actions = n_actions
        self.gamma = gamma

        # We user huber loss
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.optimizer = optim.RMSprop(self.q_net.parameters())

        self.clip_val = False

    def train(self, batch, batch_size):
        # Compute non-final states and concatenate the batch elements
        # we need to do it in order to pass to the target network
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)),
                                      device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat(
            [s.to(self.device) for s in batch.next_state if s is not None], dim=0)

        # Here we extract the states, actions and rewards from the batch
        state_batch = torch.cat([s.to(self.device) for s in batch.state], dim=0)

        action_batch = torch.tensor(batch.action, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, device=self.device)

        self.q_net.train()
        state_action_values = self.q_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size, device=self.device)

        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        if self.clip_val:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.clip_val)

        self.optimizer.step()

        return loss.cpu().detach().numpy()

    def choose_action(self, state: torch.Tensor, current_episode: int, eps_decay: int,
                      initial_eps: float = 0.99, end_eps: float = 0.2):
        eps_threshold = initial_eps + (end_eps - initial_eps) * min(
            1., float(current_episode) / eps_decay)

        if self.mode == 'eval' or random.random() > eps_threshold:
            with torch.no_grad():
                state = state.to(self.device)
                self.q_net.eval()
                chosen_action = self.q_net(state).max(1)[1].view(1, 1)
                chosen_action = chosen_action.cpu().detach()
        else:
            # We take a random action
            chosen_action = torch.tensor([[random.randrange(self.n_actions)]],
                                         dtype=torch.long)

        return chosen_action

    def switch_mode(self, mode):
        self.mode = mode

    def reset(self):
        pass


class CartpoleDQNAgentSingleCoach(BaseSimulator):
    def __init__(self, env: gym.Env, agent: CartpoleDQNAgent, n_episodes=1000,
                 display='human', batch_size=32, exploration_init=0.99,
                 exploration_end=0.2, exploration_decay=500,
                 max_steps=50, show_each=1000, fps=10,
                 plot_loss=False):
        super().__init__(env, display)

        # Initialize internal variables
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
        self.__losses = []

        self.__target_update = 10
        self.__memory_size = 10000
        self.__memory = CartpoleReplayMemory(self.__memory_size)

    def run(self):
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
        observation = torch.tensor([self._env.reset()],
                                   device=self.__agent.device).float()
        self.__agent.reset()
        self.__render(display, None)

        done = False

        for i in range(self.__max_steps):
            # Check if it's already over
            if done:
                self.__render(display, 'End of episode')
                break

            action = self.__agent.choose_action(observation, idx,
                                                eps_decay=self.__exploration_decay,
                                                initial_eps=self.__exploration_init,
                                                end_eps=self.__exploration_end)

            # Perform last action
            next_observation, reward, done, _ = self._env.step(action.item())
            reward = torch.tensor([reward], device=self.__agent.device)

            if done:
                next_observation = None
            else:
                next_observation = torch.tensor([next_observation], device=self.__agent.device).float()

            # Store the transition in memory
            self.__memory.push(observation, action, next_observation, reward)

            # Next state
            observation = next_observation

            if len(self.__memory) >= self.__batch_size:
                sample = self.__memory.sample(self.__batch_size)
                batch = CartpoleSimulation(*zip(*sample))
                loss = self.__agent.train(batch, self.__batch_size)

                if self.__plot_loss:
                    self.__losses.append(loss)
                if verbose:
                    print('Episode[{}/{}], Loss: {:.4f}, Buffer state[{}/{}], Reward: {}'
                          .format(idx + 1, self.__n_episodes, loss, len(self.__memory),
                                  self.__memory_size, reward.item()))

            # Render
            self.__render(display, (idx, i, float(reward)))

        # Update or not the targetNet
        if (idx + 1) % self.__target_update == 0:
            self.__agent.target_net.train()
            self.__agent.target_net.load_state_dict(self.__agent.q_net.state_dict())
            self.__agent.target_net.eval()

    def __render(self, display, info):
        if display is not None:
            print('\033c')
            print(info)
            self._env.render(mode=display)

    def show_loss(self):
        plt.figure()
        running_mean = np.convolve(self.__losses, np.ones(100) / 100, mode="valid")
        plt.plot(np.arange(len(running_mean)), running_mean)
        plt.savefig('loss.png')


if __name__ == '__main__':
    env = gym.make('CartPole-v0').unwrapped
    env.reset()

    agent = CartpoleDQNAgent(n_actions=2, lr=1e-3, gamma=0.999)
    coach = CartpoleDQNAgentSingleCoach(env, agent, n_episodes=500, max_steps=1000,
                                        exploration_init=0.9, exploration_end=0.05,
                                        exploration_decay=20, show_each=50,
                                        batch_size=128, plot_loss=True)
    coach.run()

    agent.save('./', 'dql_agent')
    coach.show_loss()
