from .utils import Transition, game_rewards
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
        # Array for Q value after each episode
        self.q_value_after_episode = []
        self.loss_after_episode = []
        self.steps_done = 0
        # set all rewarded events to 0 
        self.rewarded_event_counts = dict.fromkeys(game_rewards, 0)
        self.game_rewards = game_rewards
        self.shortest_path_to_coin = float("inf")
        self.shortest_path_to_enemy = float("inf")
        self.shortest_path_to_crate = float("inf")
        self.shortest_path_out_of_explosion_zone = float("inf")
        self.rewards_after_round = []
        self.rewards_of_round = []
        self.round = 0

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
