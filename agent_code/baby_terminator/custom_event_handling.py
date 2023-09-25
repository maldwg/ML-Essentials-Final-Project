import numpy as np
from agent_code.baby_terminator.path_finding import astar
import events as e
from . import custom_events as c
from .utils import *


def custom_game_events(self, old_game_state, new_game_state, events, self_action):
    """
    Generate custom events based on game states and actions.

    :param self: The agent object
    :param old_game_state: Previous game state
    :param new_game_state: Current game state
    :param events: List of events occurred
    :param self_action: The action taken by the agent
    :return: List of custom events
    """
    custom_events = []
    in_old_explosion_zone = False
    in_new_explosion_zone = False

    # if new is none something went wrong
    if new_game_state is None:
        return custom_events

    agent_x, agent_y = agent_position(new_game_state)

    # append only the events that can also be calculated for the last step
    if not_killed_by_own_bomb(events):
        custom_events.append(c.NOT_KILLED_BY_OWN_BOMB)

    # if old is none --> Last round occured
    # append all events that can be calculated in the steps before the last one
    if old_game_state is not None:
        if unallowed_bomb(self_action, old_game_state):
            custom_events.append(c.UNALLOWED_BOMB)

        old_agent_pos = agent_position(old_game_state)
        new_agent_pos = agent_position(new_game_state)
        agent_moved = has_agent_moved(old_game_state, new_game_state)

        # Check if agent took one of the optimal paths towards a coin
        if moved_towards_coin(self, old_game_state, new_game_state):
            custom_events.append(c.MOVED_TOWARDS_COIN)
        elif agent_moved and len(self.memory.shortest_paths_to_coin) > 0:
            custom_events.append(c.MOVED_AWAY_FROM_COIN)

        # update paths to all coins
        update_coin_paths(self, new_game_state)

        # check if there are bombs on the filed, if not skip calculations
        if old_game_state["bombs"]:
            in_old_explosion_zone = any(
                [
                    old_agent_pos in explosion_zones(old_game_state["field"], bomb_pos)
                    for bomb_pos, _ in old_game_state["bombs"]
                ]
            )

        if new_game_state["bombs"]:
            in_new_explosion_zone = any(
                [
                    new_agent_pos in explosion_zones(new_game_state["field"], bomb_pos)
                    for bomb_pos, _ in new_game_state["bombs"]
                ]
            )

        if not in_old_explosion_zone and in_new_explosion_zone and agent_moved:
            custom_events.append("ENTERED_POTENTIAL_EXPLOSION_ZONE")
        elif in_old_explosion_zone and not in_new_explosion_zone and agent_moved:
            custom_events.append("LEFT_POTENTIAL_EXPLOSION_ZONE")
            self.memory.left_explosion_zone = True
            # set to inf since now the shortest path is not available anymore since we are not in an explosion radius
            self.memory.shortest_paths_out_of_explosion = []

        potential_explosions = get_potential_explosions(new_game_state)
        # calculate astar to the shortest way out of explosion zone
        if (agent_x, agent_y) in potential_explosions:
            if moved_towards_end_of_explosion(self, old_game_state, new_game_state):
                custom_events.append(c.MOVED_TOWARDS_END_OF_EXPLOSION)
            # if agent did not move towards end of explosion, he might did an invalid action or a bad action
            # check if entered not in events, since otherwise double penalty for entering the zone
            # check if bomb dropped in last action, otherwise don't penalize
            elif (
                c.ENTERED_POTENTIAL_EXPLOSION_ZONE not in custom_events
                and e.BOMB_DROPPED not in events
            ):
                custom_events.append(c.STAYED_IN_EXPLOSION_RADIUS)
            update_paths_out_of_explosion(self, new_game_state)

        # check if bomb was placed so that enemy can be hit
        if e.BOMB_DROPPED in events:
            # check if enemy in explosion radius
            potential_explosion = explosion_zones(
                new_game_state["field"], agent_position(new_game_state)
            )
            for agent in new_game_state["others"]:
                if agent[-1] in potential_explosion:
                    events.append(c.ATTACKED_ENEMY)
            for x, y in potential_explosion:
                if new_game_state["field"][x, y] == 1:
                    events.append(c.CRATE_IN_EXPLOSION_ZONE)

        if e.BOMB_DROPPED in events:
            # since the agent dropped the bomb the bomb is ath the agents position
            bomb_position = (agent_x, agent_y)
            if agent_trapped_by_explosion(self, new_game_state["field"], bomb_position):
                custom_events.append(c.GUARANTEED_SUICIDE)

    return custom_events


def agent_trapped_by_explosion(self, field, bomb_position):
    """
    Checks if the agent is trapped by an explosion.

    :param self: The agent object
    :param field: The game field
    :param bomb_position: The position of the bomb
    :return: True if trapped, False otherwise
    """
    explosion_tiles = explosion_zones(field, bomb_position)
    x, y = bomb_position

    for left in range(1, 5):
        if field[x - left, y] in [-1, 1]:
            break
        if free_neighbour(self, field, (x - left, y), explosion_tiles):
            return False
    for right in range(1, 4):
        if field[x + right, y] in [-1, 1]:
            break
        if free_neighbour(self, field, (x + right, y), explosion_tiles):
            return False
    for up in range(1, 4):
        if field[x, y - up] in [-1, 1]:
            break
        if free_neighbour(self, field, (x, y - up), explosion_tiles):
            return False
    for down in range(1, 4):
        if field[x, y + down] in [-1, 1]:
            break
        if free_neighbour(self, field, (x, y + down), explosion_tiles):
            return False
    return True


def free_neighbour(self, field, position, explosion_tiles):
    """
    Checks if a tile has a free neighbour.

    :param self: The agent object
    :param field: The game field
    :param position: The position to check
    :param explosion_tiles: Tiles affected by explosion
    :return: True if has a free neighbour, False otherwise
    """
    x, y = position
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            if dx * dy == 0:
                if (
                    field[x + dx, y + dy] == 0
                    and (x + dx, y + dy) not in explosion_tiles
                ):
                    return True
    return False


def free_tiles_beneath_explosion(self, field, potential_explosions):
    """
    Find free tiles that are not affected by the explosion.

    :param self: The agent object
    :param field: The game field
    :param potential_explosions: Tiles potentially affected by explosions
    :return: List of free tiles
    """
    neighbours = []
    for x, y in potential_explosions:
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx * dy == 0:
                    # check that the neighbour is not in the explosion radius or wall or crate
                    if (x + dx, y + dy) not in potential_explosions and field[
                        x + dx, y + dy
                    ] == 0:
                        neighbours.append((x + dx, y + dy))
    return neighbours


def explosion_zones(field, bomb_pos):
    """
    Get the tiles affected by a bomb explosion.

    :param field: The game field
    :param bomb_pos: The position of the bomb
    :return: List of affected tiles
    """
    x, y = bomb_pos
    # the position of the bomb is also an explosion so add it directly
    zones = [(x, y)]
    # Add tiles for each direction until the explosion radius is reached
    for left in range(1, 4):
        if field[x - left, y] == -1:
            break
        zones.append((x - left, y))
    for right in range(1, 4):
        if field[x + right, y] == -1:
            break
        zones.append((x + right, y))
    for up in range(1, 4):
        if field[x, y - up] == -1:
            break
        zones.append((x, y - up))
    for down in range(1, 4):
        if field[x, y + down] == -1:
            break
        zones.append((x, y + down))
    return zones


def update_paths_out_of_explosion(self, new_game_state):
    """
    Update the paths to escape an explosion.

    :param self: The agent object
    :param new_game_state: Current game state
    """
    paths_out_of_explosion = get_all_paths_out_of_explosions(self, new_game_state)

    if len(paths_out_of_explosion):
        # min -1 because astar path contains start position
        path_lengths = np.array([len(path) - 1 for path in paths_out_of_explosion])
        minimum_path_length = path_lengths.min()
        minimum_path_indices = np.where(path_lengths == minimum_path_length)
        shortest_paths_out_of_explosion = [
            paths_out_of_explosion[i] for i in minimum_path_indices[0]
        ]

        if len(self.memory.shortest_paths_out_of_explosion) == 0:
            self.memory.shortest_paths_out_of_explosion = (
                shortest_paths_out_of_explosion
            )
        if (
            len(self.memory.shortest_paths_out_of_explosion[0]) - 1
            > minimum_path_length
        ):
            self.memory.shortest_paths_out_of_explosion = (
                shortest_paths_out_of_explosion
            )


def update_coin_paths(self, new_game_state):
    """
    Update the paths to collect coins.

    :param self: The agent object
    :param new_game_state: Current game state
    """
    paths_to_coins = get_all_paths_to_coins(new_game_state)
    # check if there is a coin reachable
    if len(paths_to_coins):
        # len - 1 because the starting point is always included in the path!
        path_lengths = np.array([len(path) - 1 for path in paths_to_coins])
        minimum_path_length = path_lengths.min()
        minimum_path_indices = np.where(path_lengths == minimum_path_length)
        shortest_paths_to_coin = [paths_to_coins[i] for i in minimum_path_indices[0]]

        # update memory
        self.memory.shortest_paths_to_coin = shortest_paths_to_coin


def get_all_paths_out_of_explosions(self, new_game_state):
    """
    Get all possible paths to escape explosions.

    :param self: The agent object
    :param new_game_state: Current game state
    :return: List of paths to escape explosions
    """
    agent_x, agent_y = agent_position(new_game_state)

    potential_explosions = get_potential_explosions(new_game_state)
    neighbour_tiles_out_of_explosion = free_tiles_beneath_explosion(
        self, new_game_state["field"], potential_explosions
    )

    paths_out_of_explosions = [
        astar(start=(agent_x, agent_y), goal=(x, y), field=new_game_state["field"])
        for x, y in neighbour_tiles_out_of_explosion
    ]
    paths_out_of_explosions = list(
        filter(lambda item: item is not None, paths_out_of_explosions)
    )

    return paths_out_of_explosions


def get_all_paths_to_coins(new_game_state):
    """
    Get all possible paths to coins.

    :param new_game_state: Current game state
    :return: List of paths to coins
    """
    agent_x, agent_y = agent_position(new_game_state)
    paths_to_coins = [
        astar(start=(agent_x, agent_y), goal=(x, y), field=new_game_state["field"])
        for x, y in new_game_state["coins"]
    ]
    # filter all nones (paths that are blocked)
    paths_to_coins = list(filter(lambda item: item is not None, paths_to_coins))
    return paths_to_coins


def get_potential_explosions(new_game_state):
    """
    Get tiles that could be affected by a bomb explosion.

    :param new_game_state: Current game state
    :return: List of potentially affected tiles
    """
    potential_explosions = []
    for (x, y), t in new_game_state["bombs"]:
        potential_explosions.extend(explosion_zones(new_game_state["field"], (x, y)))
    return potential_explosions


def moved_towards_end_of_explosion(self, old_game_state, new_game_state):
    """
    Check if the agent moved towards the end of an explosion.

    :param self: The agent object
    :param old_game_state: Previous game state
    :param new_game_state: Current game state
    :return: True if moved towards end, False otherwise
    """
    agent_x, agent_y = agent_position(new_game_state)
    if (
        has_agent_moved(old_game_state, new_game_state)
        and len(self.memory.shortest_paths_out_of_explosion) > 0
    ):
        for path in self.memory.shortest_paths_out_of_explosion:
            if (agent_x, agent_y) == path[1]:
                return True
    return False


def moved_towards_coin(self, old_game_state, new_game_state):
    """
    Check if the agent moved towards a coin.

    :param self: The agent object
    :param old_game_state: Previous game state
    :param new_game_state: Current game state
    :return: True if moved towards coin, False otherwise
    """
    agent_pos = agent_position(new_game_state)
    if (
        has_agent_moved(old_game_state, new_game_state)
        and len(self.memory.shortest_paths_to_coin) > 0
    ):
        for path in self.memory.shortest_paths_to_coin:
            if agent_pos == path[1]:
                return True
    return False


def not_killed_by_own_bomb(events):
    """
    Check if the agent was not killed by their own bomb.

    :param events: List of events occurred
    :return: True if not killed by own bomb, False otherwise
    """
    return e.BOMB_EXPLODED in events and not e.KILLED_SELF in events


def unallowed_bomb(self_action, old_game_state):
    """
    Check if the agent placed an unallowed bomb.

    :param self_action: The action taken by the agent
    :param old_game_state: Previous game state
    :return: True if unallowed, False otherwise
    """
    return self_action == "BOMB" and old_game_state["self"][2] == False


def has_agent_moved(old_game_state, new_game_state):
    """
    Check if the agent has moved between states.

    :param old_game_state: Previous game state
    :param new_game_state: Current game state
    :return: True if agent moved, False otherwise
    """
    return agent_position(old_game_state) != agent_position(new_game_state)


def agent_position(game_state):
    """
    Get the current position of the agent.

    :param game_state: Current game state
    :return: Tuple representing the agent's position
    """
    return game_state["self"][-1]
