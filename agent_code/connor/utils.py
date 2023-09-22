import torch
from collections import namedtuple
import events as e
from . import custom_events as c
import numpy as np

DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
device = torch.device("cpu")

game_rewards = {
    # long term goal
    e.SURVIVED_ROUND: 100,
    # killing goals
    e.KILLED_OPPONENT: 100,
    e.OPPONENT_ELIMINATED: -2,
    e.KILLED_SELF: -170,
    e.GOT_KILLED: -100,
    e.INVALID_ACTION: -10,
    # additional penalty when laying 2 bombs in a row
    c.UNALLOWED_BOMB: -10,
    # coin goals
    e.COIN_FOUND: 7.5,
    e.COIN_COLLECTED: 50,
    c.MOVED_TOWARDS_COIN: 2.5,
    # crate goals
    e.CRATE_DESTROYED: 7.5,
    c.CRATE_IN_EXPLOSION_ZONE: 10,
    c.NOT_KILLED_BY_OWN_BOMB: 7.5,
    c.GUARANTEED_SUICIDE: -120,
    # bomb related goals
    c.MOVED_TOWARDS_END_OF_EXPLOSION: 5,
    c.LEFT_POTENTIAL_EXPLOSION_ZONE: 12,
    c.ENTERED_POTENTIAL_EXPLOSION_ZONE: -5,
    c.ATTACKED_ENEMY: 30,
    e.MOVED_DOWN: -2,
    e.MOVED_LEFT: -2,
    e.MOVED_RIGHT: -2,
    e.MOVED_UP: -2,
    e.WAITED: -10,
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
    for event in events:
        if event in self.memory.rewarded_event_counts:
            self.memory.rewarded_event_counts[event] += 1


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

    position += action_dict[action]
    empty_spot = (
        not (field[position[0], position[1]] in [-1, 1])
        and not ((position[0], position[1]) in bomb_positions)
        and not ((position[0], position[1]) in enemy_positions)
    )
    valid_bomb_dropped = bomb_allowed if action == "BOMB" else True

    return empty_spot and valid_bomb_dropped
