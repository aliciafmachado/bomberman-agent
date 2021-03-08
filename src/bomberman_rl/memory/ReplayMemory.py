# Replay memory, so that we store the previous simulations

import random
from collections import namedtuple

Simulation = namedtuple('Simulation',
                        ('state', 'action', 'next_state', 'reward', 'time', 'next_time'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a simulation."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Simulation(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)