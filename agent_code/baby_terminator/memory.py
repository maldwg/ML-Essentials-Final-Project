from .utils import Transition
import random

class ReplayMemory:
    """
    Class to store the states in on which the agent will be able to learn.
    In this memory the proposed Transitions of 

    Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

    will be pushed
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
