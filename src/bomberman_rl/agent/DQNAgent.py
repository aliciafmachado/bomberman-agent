# Based on the Tutorial on DQNs from Pytorch
# Link to tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

# Training pipeline
import torch
import random
import numpy as np
import torch.optim as optim
from bomberman_rl.agent.DQNModel import DQNModel
import torchvision.transforms as transforms

class DQNAgent():
    '''
    Deep Q-Learning Agent for the Bomberman environment
    '''

    def __init__(self, height, width, n_dim=5, n_actions=6, 
        target_update=10, lr=1e-3, gamma=0.9):
        '''
        @param height: height of the environemnt frame
        @param width: width of the environment frame
        @param n_dim: Number of frames for each state
        @param n_actions: Number of possible actions
        @param target_update: after how many episodes of training should we
            update targetNet
        @param lr: learning rate
        '''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.targetNet = DQNModel(height, width, n_dim=n_dim, device=self.device)
        self.qNet = DQNModel(height, width, n_dim=n_dim, device=self.device)
        self.n_actions = n_actions
        self.gamma = gamma

        # Temporary variable
        self.time = 0
        self.max_time = 6

        # The update frequence of the target net
        self.target_update = target_update

        # We user huber loss
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.qNet.parameters(), lr)

        # TODO: implement how to get to clip or not the weights
        self.clip_val = True

    def train(self, batch, batch_size):
        '''
        @param batch: sample of the memory
        @param batch_size: size of the batch
        @return mean loss for the batch
        '''

        transform = transforms.ToTensor()

        # Compute non-final states and concatenate the batch elements
        # we need to do it in order to pass to the target network
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=self.device, dtype=torch.bool)

        non_final_next_states = torch.cat([transform(np.array(s)).to(self.device).unsqueeze(0) for s in batch.next_state
                                                    if s is not None], 0).type(torch.FloatTensor)

        # Here we extract the states, actions and rewards from the batch
        state_batch = torch.cat([transform(np.array(s)).to(self.device).unsqueeze(0) for s in 
            batch.state], 0).type(torch.FloatTensor)
        action_batch = torch.tensor(batch.action, device=self.device).unsqueeze(0)
        reward_batch = torch.tensor(batch.reward, device=self.device).unsqueeze(0)
        time_batch = torch.tensor(batch.time, device=self.device).unsqueeze(1) # TODO: fix this

        # First we calculate the Q(s_t, a) for the actions taken
        # so that we get the value that we would get from the state-action
        # in the batch
        state_action_values = self.qNet(state_batch, time_batch).gather(1, action_batch)

        # We compute the values for all next states using the target net
        # and then we use the bellman
        # equation formulation to find the expected state action values of 
        # the current state
        # We initialize the values and use the previous mask so that
        # the final states get state value equal 0 and the other ones that are
        # not final get their correct values
        next_state_values = torch.zeros(batch_size, device=self.device)
        next_state_values[non_final_mask] = self.targetNet(non_final_next_states, time_batch).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute huber loss
        loss = self.loss_fn(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        if self.clip_val:
            # Put all grads between -1 and 1
            torch.nn.utils.clip_grad_norm_(self.qNet.parameters(),
                self.clip_val)

        self.optimizer.step()

        return loss.detach().numpy()        
    
    def select_action(self, state, time, current_episode, eps_decay=200., initial_eps=0.99,
        end_eps=0.25):
        '''
        We select actions by taking either random actions or by
            taking the actions considering the value returned by the qNet
        The threshold varies linearly depending on how many episodes did we train
        @param: state: the state we currently are
        @param initial_eps: the initial threshold
        @param end_eps: the ending threshold
        @param eps_decay: how we calculate the fraction
        '''
        sample = random.random()
        
        # We get the current threshold to choose an action randomly
        # or to choose an action from the network
        eps_threshold = initial_eps + (end_eps - initial_eps) * \
            min(1., float(current_episode) / eps_decay)

        if sample > eps_threshold:
            with torch.no_grad():
                # We get the action that have the greatest value among
                # all the ones calculated by the neural network
                chosen_action = self.qNet(state, time).max(1)[1].view(1, 1) 

        else:
            # We take a random action
            chosen_action = torch.tensor([[random.randrange(self.n_actions)]], 
                device=self.device, dtype=torch.long)

        if chosen_action == 5:
            self.time = self.max_time

        else:
            self.time = max(self.time - 1, 0)

        return chosen_action

    def evaluate():
        self.qNet.eval()
