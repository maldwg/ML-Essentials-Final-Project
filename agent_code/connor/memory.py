from .utils import game_rewards
from . import custom_events as c
import random


class ReplayMemory:
    """
    """

    def __init__(self):
        self.steps_done = 0
        self.loss_after_episode = []
        self.rewarded_event_counts = dict.fromkeys(game_rewards, 0)
        self.game_rewards = game_rewards
        self.shortest_paths_to_coin = []
        self.shortest_paths_to_enemy = []
        self.shortest_paths_to_crate = []
        self.shortest_paths_out_of_explosion = []
        self.left_explosion_zone = False
        self.rewards_after_round = []
        self.rewards_of_round = []

        self.step_action_rewards = []
        self.episode_action_rewards = []

    def __len__(self):
        """
        return length based on played episodes
        """
        return len(self.rewards_after_round)
