# Run main for DQNAgent

from bomberman_rl.rl.ReplayMemory import ReplayMemory
from bomberman_rl.rl.ReplayMemory import Simulation

MEMORY_SIZE=10000

# TODO
# Return if the there are not experiences enough to train
# the agent
# if(len(self.memory)) < batch_size:
#     return

# sample = self.memory.sample(batch_size)

#         # We transpose the simulations in order to get 
# # the actions and rewards in the standart way
# batch = Simulation(*zip(*sample))

# self.memory = ReplayMemory(MEMORY_SIZE)