# Based on the Tutorial on DQNs from Pytorch
# Link to tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

# Training pipeline
import torch
import torch.optim as optim
from bomberman_rl.rl.DQNModel import DQNModel

class DQNAgent():
    '''
    Deep Q-Learning Agent for the Bomberman environment
    '''

    def __init__(self, height, width, n_dim=5, n_actions=6, 
        target_update=10, lr=1e-3, self.gamma=0.9):
        '''
        @param height: height of the environemnt frame
        @param width: width of the environment frame
        @param n_dim: Number of frames for each state
        @param n_actions: Number of possible actions
        @param target_update: after how many episodes of training should we
            update targetNet
        @param lr: learning rate
        '''
        self.targetNet = DQNModel(height, width, n_dim=n_dim)
        self.qNet = DQNModel(height, width, n_dim=n_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_actions = n_actions
        
        # Number of steps of training already done
        self.current_episode = 0

        # The update frequence of the target net
        self.target_update = target_update

        # We user huber loss
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.qNet.parameters(), lr)

        # TODO: implement how to get to clip or not the weights
        self.clip_value = False

    def train(self, batch, batch_size):
        '''
        @param batch: sample of the memory
        @param batch_size: size of the batch
        @return mean loss for the batch
        '''
        self.current_episode += 1

        # Compute non-final states and concatenate the batch elements
        # we need to do it in order to pass to the target network
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])

        # Here we extract the states, actions and rewards from the batch
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # First we calculate the Q(s_t, a) for the actions taken
        # so that we get the value that we would get from the state-action
        # in the batch
        state_action_values = qNet(state_batch).gather(1, action_batch)

        # We compute the values for all next states using the target net
        # and then we use the bellman
        # equation formulation to find the expected state action values of 
        # the current state
        # We initialize the values and use the previous mask so that
        # the final states get state value equal 0 and the other ones that are
        # not final get their correct values
        next_state_values = torch.zeros(batch_size, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute huber loss
        loss = self.loss_fn(state_action_values, expected_state_action_values)

        # Optimize the model
        sellf.optimizer.zero_grad()
        loss.backward()

        if self.clip_value:
            # Put all grads between -1 and 1
            torch.nn.utils.clip_grad_norm_(self.qNet.parameters(),
                self.clip_val)

        self.optimizer.step()

        # Update or not the targetNet
        if self.current_episode % self.target_update == 0: 
            target_net.load_state_dict(qNet.state_dict())

        return loss.detach().numpy()        
    
    def select_action(self, state, initial_eps=0.9, 
        end_eps=0.05, eps_decay=200.):
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
        eps_threshold = initial_eps + (end_eps - initial_eps) * 
            np.max(1., float(self.current_episode) / eps_decay))

        if sample > eps_threshold:
            with torch.no_grad():
                # We get the action that have the greatest value among
                # all the ones calculated by the neural network
                return self.qNet(state).max(1)[1].view(1, 1)

        else:
            # We take a random action
            return torch.tensor([[random.randrange(self.n_actions)]], 
                device=self.device, dtype=torch.long)
        
