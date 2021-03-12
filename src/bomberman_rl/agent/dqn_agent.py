import torch
import random
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

    def __init__(self, height, width, n_dim=5, n_actions=6, lr=1e-3, gamma=0.9):
        """
        @param height: height of the environment frame
        @param width: width of the environment frame
        @param n_dim: Number of frames for each state
        @param n_actions: Number of possible actions
        @param lr: learning rate
        """
        super().__init__()

        # Temporary variable
        self.time = 0
        self.max_time = 9
        self.mode = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_net = DQNModel(height, width, n_dim=n_dim,
                                   time_size=self.max_time + 1, device=self.device)
        self.q_net = DQNModel(height, width, n_dim=n_dim, time_size=self.max_time + 1,
                              device=self.device)
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

        # Compute non-final states and concatenate the batch elements
        # we need to do it in order to pass to the target network
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)),
                                      device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat(
            [s.to(self.device) for s in batch.next_state if s is not None], dim=0)

        # Here we extract the states, actions and rewards from the batch
        state_batch = torch.cat([s.to(self.device) for s in batch.state], dim=0)

        action_batch = torch.tensor(batch.action, device=self.device).unsqueeze(0)
        reward_batch = torch.tensor(batch.reward, device=self.device).unsqueeze(0)
        time_batch = torch.tensor(batch.time, device=self.device)
        next_time_batch = torch.tensor(batch.next_time, device=self.device)

        # First we calculate the Q(s_t, a) for the actions taken
        # so that we get the value that we would get from the state-action
        # in the batch
        self.q_net.train()
        state_action_values = self.q_net(state_batch, time_batch).gather(1, action_batch)

        # We compute the values for all next states using the target net
        # and then we use the bellman
        # equation formulation to find the expected state action values of 
        # the current state
        # We initialize the values and use the previous mask so that
        # the final states get state value equal 0 and the other ones that are
        # not final get their correct values
        next_state_values = torch.zeros(batch_size, device=self.device)

        # We decrease the timer for the next state
        next_state_values[non_final_mask] = \
            self.target_net(non_final_next_states, next_time_batch).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute huber loss
        loss = self.loss_fn(state_action_values, expected_state_action_values)

        # Optimize the model
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

        # We get the current threshold to choose an action randomly
        # or to choose an action from the network
        eps_threshold = initial_eps + (end_eps - initial_eps) * min(
            1., float(current_episode) / eps_decay)

        if self.mode == 'eval' or random.random() > eps_threshold:
            with torch.no_grad():
                state = state.to(self.device)
                time = torch.tensor([self.time], device=self.device)

                # We get the action that have the greatest value among
                # all the ones calculated by the neural network
                self.q_net.eval()
                chosen_action = self.q_net(state, time).max(1)[1].view(1, 1)
                chosen_action = chosen_action.cpu().detach()
        else:
            # We take a random action
            chosen_action = torch.tensor([[random.randrange(self.n_actions)]],
                                         dtype=torch.long)

        if chosen_action == PLACE_BOMB:
            self.time = self.max_time
        else:
            self.time = max(self.time - 1, 0)

        return chosen_action

    def switch_mode(self, mode):
        self.mode = mode

    def reset(self):
        self.time = 0
