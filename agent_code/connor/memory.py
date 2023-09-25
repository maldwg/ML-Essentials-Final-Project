from .utils import game_rewards


class ReplayMemory:
    """
    A class for storing and managing replay memory for reinforcement learning.

    Attributes:
        steps_done (int): The total number of steps taken across all episodes.
        loss_after_episode (list): A list to store loss values after each episode.
        rewarded_event_counts (dict): A dictionary to keep track of the count of each rewarded event.
        game_rewards (dict): A dictionary that contains the rewards for various game events, defined in utils.py.
        shortest_paths_to_coin (list): A list to store the shortest paths to coins in each episode.
        shortest_paths_to_enemy (list): A list to store the shortest paths to enemies in each episode.
        shortest_paths_to_crate (list): A list to store the shortest paths to crates in each episode.
        shortest_paths_out_of_explosion (list): A list to store the shortest paths to escape an explosion.
        left_explosion_zone (bool): A flag to indicate whether the agent has left the explosion zone.
        rewards_after_round (list): A list to store rewards after each round.
        rewards_of_round (list): A list to store the rewards of the current round.
        step_action_rewards (list): A list to store the rewards for each action taken in the current episode.
        episode_action_rewards (list): A list to store action and corresponding rewards for each episode.
    """

    def __init__(self):
        """
        Initializes a new ReplayMemory instance.
        """
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
        Returns the length based on the number of played episodes.

        :return int: The number of episodes stored in replay memory.
        """
        return len(self.rewards_after_round)
