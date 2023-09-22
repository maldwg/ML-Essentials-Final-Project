import torch
from collections import namedtuple
import events as e
from . import custom_events as c
import functools
import numpy as np
import math
import json

DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
device = torch.device("cpu")
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


def z_normalize_rewards(rewards: dict) -> dict:
    """
    Z-normalizes a dictionary of rewards using the standard score (z-score).

    Args:
        rewards (dict): A dictionary containing rewards.

    Returns:
        dict: A dictionary of rewards with values normalized using the z-score.

    Example:
        >>> rewards = {
        ...     'reward_1': 100,
        ...     'reward_2': 200,
        ...     'reward_3': 300
        ... }
        >>> z_normalized = z_normalize_rewards(rewards)
    """
    # Extract the reward values
    reward_values = list(rewards.values())

    # Calculate mean and standard deviation
    mean = np.mean(reward_values)
    std_dev = np.std(reward_values)

    # Normalize the rewards using z-score
    normalized_rewards = {
        key: (value - mean) / std_dev for key, value in rewards.items()
    }

    return normalized_rewards


game_rewards_not_normalized = {
    # long term goal
    e.SURVIVED_ROUND: 100,
    c.SCORE_REWARD: 20,
    c.PLACEMENT_REWARD: 150,
    # killing goals
    # killing an opponent should give less points than killing yourself otherwise the agent will suicide bomb an enemy
    e.KILLED_OPPONENT: 100,
    e.OPPONENT_ELIMINATED: -2,
    e.KILLED_SELF: -50,
    e.GOT_KILLED: -30,
    # correct actions
    e.INVALID_ACTION: -10,
    # additional penalty when laying 2 bombs in a row
    c.UNALLOWED_BOMB: -10,
    # coin goals
    e.COIN_FOUND: 7.5,
    e.COIN_COLLECTED: 50,
    c.MOVED_TOWARDS_COIN: 2.5,
    # crate goals
    # crate destroyed im verhältnis zu coin found ändern, ggf. mehr für coin found als crate destroyed
    # little bit smaller since it is delayed --> adds up with the bomb_before_crate signal
    e.CRATE_DESTROYED: 7.5,
    c.CRATE_IN_EXPLOSION_ZONE: 10,
    c.NOT_KILLED_BY_OWN_BOMB: 7.5,
    c.GUARANTEED_SUICIDE: -100,
    # bomb related goals
    c.MOVED_TOWARDS_END_OF_EXPLOSION: 5,
    c.LEFT_POTENTIAL_EXPLOSION_ZONE: 10,
    c.ENTERED_POTENTIAL_EXPLOSION_ZONE: -5,
    c.ATTACKED_ENEMY: 30,
    # penalize default actions otherwise too many watis and random moves
    # 1.5 because this might prevent trembling since then the reward for going left + right + left is negative even if 2 times the agent moved to a coin
    e.MOVED_DOWN: -1.5,
    e.MOVED_LEFT: -1.5,
    e.MOVED_RIGHT: -1.5,
    e.MOVED_UP: -1.5,
    e.WAITED: -7.5,
    # Only give points if enemy is attacked or crate is in explosion zone
    e.BOMB_DROPPED: -10,
}


def increment_event_counts(self, events):
    """
    Increments the count of each event in the memory based on the provided events list.

    Args:
        self: The object instance.
        events (list): List of events to be counted.

    Returns:
        None. Updates the event counts in the memory in-place.
    """
    # self.logger.info("Increment count of events in memory")
    for event in events:
        if event in self.memory.rewarded_event_counts:
            self.memory.rewarded_event_counts[event] += 1
    # self.logger.info(f"incremented events: {self.memory.rewarded_event_counts}")


# Helper function to check if a position is blocked by walls or crates
def is_blocked(position, field):
    """
    Determines if a given position is blocked by walls or crates in the game state.

    Args:
        position (tuple): A tuple representing the x and y coordinates.
        game_state (dict): The current game state.

    Returns:
        bool: True if the position is blocked, False otherwise.
    """
    x, y = position
    return field[x, y] == -1 or field[x, y] == 1


def is_action_valid(self, state, action):
    """
    Checks if a given action is valid based on the current state of the game.

    Args:
        self: The object instance.
        state (dict): The current game state.
        action (str): The action to be checked ('UP', 'DOWN', 'RIGHT', 'LEFT', 'WAIT', 'BOMB').

    Returns:
        bool: True if the action is valid in the current state, False otherwise.
    """
    field = state["field"]

    bomb_positions = [pos[0] for pos in state["bombs"]]
    enemy_positions = [pos[0] for pos in state["others"]]

    bomb_allowed = state["self"][2]
    position = np.array(state["self"][-1])

    action_dict = {
        "UP": np.array([0, -1]),
        "DOWN": np.array([0, 1]),
        "RIGHT": np.array([1, 0]),
        "LEFT": np.array([-1, 0]),
        "BOMB": np.array([0, 0]),
        "WAIT": np.array([0, 0]),
    }

    # check positions
    position += action_dict[action]
    empty_spot = (
        not (field[position[0], position[1]] in [-1, 1])
        and not ((position[0], position[1]) in bomb_positions)
        and not ((position[0], position[1]) in enemy_positions)
    )
    valid_bomb_dropped = bomb_allowed if action == "BOMB" else True

    return empty_spot and valid_bomb_dropped


def calculate_eps_threshold(self, EPS_START, EPS_END, EPS_DECAY):
    return EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * self.memory.steps_done / EPS_DECAY
    )


def read_hyperparameters():
    with open("./parameters.json", "r") as f:
        hyperparameters = json.load(f)
    return hyperparameters
