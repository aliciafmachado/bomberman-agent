import torch
import random
import numpy as np
from torch import optim

from bomberman_rl.agent.trainable_agent import TrainableAgent
from bomberman_rl.agent.dqn_model import DQNModel
from bomberman_rl.envs.conventions import PLACE_BOMB


class DQNAgent(TrainableAgent):
    """
    Deep Q-Learning Agent for the Bomberman environment.

    Based on the Tutorial on DQNs from Pytorch
    Link to tutorial:
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """

    def __init__(self, observation_shape, n_actions, lr=1e-3, gamma=0.9,
                 temporal_mode="frame_stack", nb_frames=4):
        """
        @param observation_shape: shape the environment frame
        @param n_actions: Number of possible actions
        @param lr: learning rate
        """
        super().__init__()
        self.mode = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if temporal_mode == "time" or temporal_mode == None:
            self.nb_frames = 1

        elif temporal_mode == "frame_stack":
            self.nb_frames = nb_frames

        else:
            raise ValueError('Invalid temporal mode: {:}. Please use frame-stack, time or None.'
                .format(temporal_mode))

        # Calculate frame stack shape
        frame_stack_shape = np.array(observation_shape)
        frame_stack_shape[-1] = frame_stack_shape[-1] * self.nb_frames

        # Temporary variable
        self.time_size = 10
        self.time = torch.tensor([0], device=self.device)
        self.max_time = torch.tensor([self.time_size - 1], device=self.device)

        self.target_net = DQNModel(frame_stack_shape, n_actions, time_size=self.time_size,
                                   device=self.device, temporal_mode=temporal_mode)
        self.q_net = DQNModel(frame_stack_shape, n_actions, time_size=self.time_size,
                              device=self.device,temporal_mode=temporal_mode)
        self.n_actions = n_actions
        self.gamma = gamma

        # We user huber loss
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr)

        self.clip_val = True

    def train(self, batch, batch_size):
        """
        @param batch: sample of the memory
        @param batch_size: size of the batch
        @return mean loss for the batch
        """

        # Compute non-final states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)),
                                      device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None])

        # Extract batch
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        time_batch = torch.cat(batch.time)
        next_time_batch = torch.cat(batch.next_time)

        self.q_net.train()
        state_action_values = self.q_net(state_batch, time_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size, device=self.device)
        next_state_values[non_final_mask] = \
            self.target_net(non_final_next_states, next_time_batch).max(1)[0].detach()

        # Compute expected Q values
        expected_state_action_values = next_state_values * self.gamma + reward_batch

        loss = self.loss_fn(state_action_values,
                            expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()

        if self.clip_val:
            # Put all grads between -1 and 1
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.clip_val)

        self.optimizer.step()

        return loss.cpu().detach().numpy()

    def choose_action(self, state: torch.Tensor, current_episode: int, eps_decay: int,
                      initial_eps: float = 0.99, end_eps: float = 0.2):
        """
        We select actions by taking either random actions or by
            taking the actions considering the value returned by the qNet
        The threshold varies linearly depending on how many episodes did we train
        @param state: the state we currently are
        @param current_episode:
        @param initial_eps: the initial threshold
        @param end_eps: the ending threshold
        @param eps_decay: how we calculate the fraction
        """

        # Linear decay
        eps_threshold = initial_eps + (end_eps - initial_eps) * min(
            1., float(current_episode) / eps_decay)

        if self.mode == 'eval' or random.random() > eps_threshold:
            with torch.no_grad():
                self.q_net.eval()
                chosen_action = self.q_net(state, self.time).max(1)[1].view(1, 1)
        else:
            chosen_action = torch.tensor([[random.randrange(self.n_actions)]],
                                         dtype=torch.long, device=self.device)

        if chosen_action == PLACE_BOMB:
            self.time = self.max_time
        else:
            self.time = torch.clamp(self.time - 1, 0, self.time_size - 1)

        return chosen_action

    def switch_mode(self, mode):
        self.mode = mode

    def reset(self):
        self.time = torch.tensor([0], device=self.device)
