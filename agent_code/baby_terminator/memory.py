from .utils import Transition, game_rewards
from . import custom_events as c
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
        """
        Initializes a new ReplayMemory object.

        :param capacity: The maximum number of transitions to store.
        
        :attribute capacity (int): The maximum capacity of the replay memory.
        :attribute memory (list): List to hold the transitions.
        :attribute position (int): The next position to insert a transition.
        :attribute q_value_after_episode (list): Array for Q value after each episode.
        :attribute loss_after_episode (list): Array for loss after each episode.
        :attribute steps_done (int): Number of steps completed.
        :attribute rewarded_event_counts (dict): Count of each rewarded event.
        :attribute game_rewards (dict): Game rewards configuration.
        :attribute game_rewards_original (dict): Original game rewards configuration.
        :attribute shortest_paths_to_coin (list): Shortest paths to coins.
        :attribute shortest_paths_to_enemy (list): Shortest paths to enemies.
        :attribute shortest_paths_to_crate (list): Shortest paths to crates.
        :attribute shortest_paths_out_of_explosion (list): Shortest paths out of explosion zones.
        :attribute left_explosion_zone (bool): Flag to indicate if agent left an explosion zone.
        :attribute rewards_after_round (list): Rewards after each round.
        :attribute rewards_of_round (list): Rewards in the current round.
        :attribute steps_since_last_update (int): Steps since the last update.
        :attribute update_frequency (int): Frequency of updates.
        :attribute train_params (NoneType): Training parameters (currently not set).
        :attribute eps_decay_params (NoneType): Parameters for epsilon decay (currently not set).
        """
        self.capacity = int(capacity)
        self.memory = []
        self.position = 0
        # Array for Q value after each episode
        self.q_value_after_episode = []
        self.loss_after_episode = []
        self.steps_done = 0
        # set all rewarded events to 0
        self.rewarded_event_counts = dict.fromkeys(game_rewards, 0)
        self.game_rewards = game_rewards
        self.game_rewards_original = game_rewards.copy()
        self.shortest_paths_to_coin = []
        self.shortest_paths_to_enemy = []
        self.shortest_paths_to_crate = []
        self.shortest_paths_out_of_explosion = []
        self.left_explosion_zone = False
        self.rewards_after_round = []
        self.rewards_of_round = []
        self.steps_since_last_update = 0
        self.update_frequency = 500
        self.train_params = None
        self.eps_decay_params = None

    def push(self, *args):
        """
        Saves a transition to the replay memory.
        
        :param args: A tuple containing the state, action, next_state, and reward.
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples a batch of transitions from the replay memory.
        
        :param batch_size: Number of transitions to sample.
        
        :return: A list of sampled transitions.
        """
        # Exclude None values from sampled data
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Returns the current size of the replay memory.
        
        :return: The length of the replay memory.
        """
        return len(self.memory)
