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
        # self.game_rewards = game_rewards
        self.shortest_paths_to_coin = []
        self.shortest_paths_to_enemy = []
        self.shortest_paths_to_crate = []
        self.shortest_paths_out_of_explosion = []
        self.rewards_after_round = []
        self.rewards_of_round = []
        self.steps_since_last_update = 0
        self.update_frequency = 500 

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # Exclude None values from sampled data
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
