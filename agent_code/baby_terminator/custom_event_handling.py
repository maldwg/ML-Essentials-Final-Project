import numpy as np
from agent_code.baby_terminator.path_finding import astar
import events as e
from . import custom_events as c


def custom_game_events(self, old_game_state, new_game_state, events, self_action):
    custom_events = []
    valid_move = e.INVALID_ACTION not in events
    in_old_explosion_zone = False
    in_new_explosion_zone = False
    # init with high value so if no bomb was in old state but one is discovered in new one, there is no penalty since 0 would be < distance to bomb
    old_distance_to_bomb = 1000
    new_distance_to_bomb = 1000
    # init with high value as safe space is set to ~3
    closest_bomb = 1000
    safe_distance = 2
    # init with high value so if no coin was in old state but one is discovered in new one, there is no penalty since 0 would be < distance to coin
    old_distance_to_coin = 1000
    new_distance_to_coin = 1000
    old_distance_to_enemy = 1000
    new_distance_to_enemy = 1000

    # if new is none something went wrong
    if new_game_state is None:
        return custom_events

    agent_x, agent_y = new_game_state["self"][-1]

    # append only the events that can also be calculated for the last step
    if not_killed_by_own_bomb(events):
        custom_events.append(c.NOT_KILLED_BY_OWN_BOMB)

    # if old is none --> Last round occured
    # append all events that can be calculated in the steps before the last one
    if old_game_state is not None:

        if unallowed_bomb(self_action, old_game_state):
            custom_events.append(c.UNALLOWED_BOMB)

        old_agent_pos = old_game_state["self"][-1]
        new_agent_pos = new_game_state["self"][-1]
        agent_moved = has_agent_moved(old_game_state, new_game_state)
        self.logger.info(f"Old agent position: {old_agent_pos}")
        self.logger.info(f"New agent position: {new_agent_pos}")

        # Check if agent took one of the optimal paths towards a coin
        if moved_towards_coin(self, old_game_state, new_game_state):
            events.append(c.MOVED_TOWARDS_COIN)

        # update paths to all coins
        update_coin_paths(self, new_game_state, events)

        potential_explosions = get_potential_explosions(new_game_state)
        # calculate astar to the shortest way out of explosion zone
        if (agent_x, agent_y) in potential_explosions:
            # self.logger.info("agent in explosion zone")
            if moved_towards_end_of_explosion(self, old_game_state, new_game_state):
                events.append(c.MOVED_TOWARDS_END_OF_EXPLOSION)
            update_paths_out_of_explosion(self, new_game_state)
        else:
            self.logger.info("agent not in explosion zone of bombs")

        # check if bomb was placed so that enemy can be hit
        # check if layed bomb
        if e.BOMB_DROPPED in events:
            # check if enemy in explosion radius
            potential_explosion = explosion_zones(new_game_state["field"], new_game_state["self"][-1])
            for agent in new_game_state["others"]:
                if agent[-1] in potential_explosion:
                    # TODO: check if it works
                    self.logger.info("attacked enemy")
                    events.append(c.ATTACKED_ENEMY)
            for (x, y) in potential_explosion:
                if new_game_state["field"][x, y] == 1:
                    self.logger.info("crate in explosion zone")
                    events.append(c.CRATE_IN_EXPLOSION_ZONE)

        # check if there are bombs on the filed, if not skip calculations
        if old_game_state["bombs"]:
            in_old_explosion_zone = any([old_agent_pos in explosion_zones(old_game_state["field"], bomb_pos) for bomb_pos, _ in old_game_state["bombs"]])

        if new_game_state["bombs"]:
            in_new_explosion_zone = any([new_agent_pos in explosion_zones(new_game_state["field"], bomb_pos) for bomb_pos, _ in new_game_state["bombs"]])

        if not in_old_explosion_zone and in_new_explosion_zone and agent_moved:
            # self.logger.info(f"EXPLOSION ZONE ENTERED: {in_new_explosion_zone} < {in_old_explosion_zone}")
            custom_events.append("ENTERED_POTENTIAL_EXPLOSION_ZONE")
        elif in_old_explosion_zone and not in_new_explosion_zone and agent_moved:
            # self.logger.info(f"EXPLOSION ZONE LEFT: {in_new_explosion_zone} < {in_old_explosion_zone}")
            custom_events.append("LEFT_POTENTIAL_EXPLOSION_ZONE")
            # set to inf since now the shortest path is not available anymore since we are not in an explosion radius
            self.memory.shortest_paths_out_of_explosion = []

    return custom_events


def tiles_beneath_explosion(self, new_game_state, potential_explosions):
    neighbours = []
    for x, y in potential_explosions:
        for dx in range(-1 , 2):
            for dy in range(-1 , 2):
                if dx * dy == 0:
                    # check that the neighbour is not in the explosion radius or wall or crate
                    if ( x + dx, y + dy) not in potential_explosions and new_game_state["field"][x+dx, y+dy] == 0:
                        neighbours.append((x+dx, y+dy))
    return neighbours



def explosion_zones(field, bomb_pos):
    """Returns a list of coordinates that will be affected by the bomb's explosion."""
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
        if field[x, y + up] == -1:
            break
        zones.append((x, y + up))
    for down in range(1, 4):
        if field[x, y - down] == -1:
            break
        zones.append((x, y - down))
    return zones


def update_paths_out_of_explosion(self, new_game_state):
    paths_out_of_explosion = get_all_paths_out_of_explosions(self, new_game_state)
    # self.logger.info(f"paths out of explosion: {paths_out_of_explosion}")

    if len(paths_out_of_explosion):
        # min -1 because astar path contains start position
        path_lengths = np.array([len(path) - 1 for path in paths_out_of_explosion])
        minimum_path_length = path_lengths.min()
        minimum_path_indices = np.where(path_lengths == minimum_path_length)
        shortest_paths_out_of_explosion = [paths_out_of_explosion[i] for i in minimum_path_indices[0]]
        self.logger.info(f"shortest path out of explosion: {shortest_paths_out_of_explosion}")
        self.logger.info(min(paths_out_of_explosion, key=len))

        if len(self.memory.shortest_paths_out_of_explosion) == 0:
            self.memory.shortest_paths_out_of_explosion = shortest_paths_out_of_explosion
        if len(self.memory.shortest_paths_out_of_explosion[0]) - 1 > minimum_path_length:
            self.memory.shorest_paths_out_of_explosion = shortest_paths_out_of_explosion


def update_coin_paths(self, new_game_state, events):
    paths_to_coins = get_all_paths_to_coins(new_game_state)
    # self.logger.info(f"paths to coins: {paths_to_coins}")
    # check if there is a coin reachable
    if len(paths_to_coins):
        # len - 1 because the starting point is always included in the path!
        path_lengths = np.array([len(path) - 1 for path in paths_to_coins])
        minimum_path_length = path_lengths.min()
        minimum_path_indices = np.where(path_lengths == minimum_path_length)
        shortest_paths_to_coin = [paths_to_coins[i] for i in minimum_path_indices[0]]
        self.logger.info(f"New shortest paths to coin: {shortest_paths_to_coin}")
        
        # update memory
        self.memory.shortest_paths_to_coin = shortest_paths_to_coin


def get_all_paths_out_of_explosions(self, new_game_state):
    agent_x, agent_y = new_game_state["self"][-1]

    potential_explosions = get_potential_explosions(new_game_state)
    neighbour_tiles_out_of_explosion = tiles_beneath_explosion(self, new_game_state, potential_explosions)

    paths_out_of_explosions = [astar(start=(agent_x, agent_y), goal=(x, y), field=new_game_state["field"]) for x, y in neighbour_tiles_out_of_explosion]
    # filter all invalid (none) paths
    paths_out_of_explosions = list(filter(lambda item: item is not None, paths_out_of_explosions))
    return paths_out_of_explosions


def get_all_paths_to_coins(new_game_state):
    agent_x, agent_y = new_game_state["self"][-1]
    paths_to_coins = [astar(start=(agent_x, agent_y), goal=(x, y), field=new_game_state["field"]) for x, y in new_game_state["coins"]]
    # filter all nones (paths that are blocked)
    paths_to_coins = list(filter(lambda item: item is not None, paths_to_coins))
    return paths_to_coins


def get_potential_explosions(new_game_state):
    potential_explosions = []
    for (x, y), t in new_game_state["bombs"]:
        potential_explosions.extend(explosion_zones(new_game_state["field"], (x, y)))
    return potential_explosions


def agent_in_front_of_crate(self, field, agent_pos):
    agent_x, agent_y = agent_pos[0], agent_pos[1]
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            # check only vertical and horizontal lines
            if dx * dy == 0:
                if field[dx + agent_x][dy + agent_y] == 1:
                    # self.logger.info(f"{dx + agent_x}, {dy + agent_y}")
                    # self.logger.info("agent before crate")
                    return True
    # self.logger.info("Agent not in front of crate")
    return False


def moved_towards_end_of_explosion(self, old_game_state, new_game_state):
    agent_x, agent_y = new_game_state["self"][-1]
    if has_agent_moved(old_game_state, new_game_state) and len(self.memory.shortest_paths_out_of_explosion) > 0:
        for path in self.memory.shortest_paths_out_of_explosion:
            if (agent_x, agent_y) == path[1]:
                return True
    return False


def moved_towards_coin(self, old_game_state, new_game_state):
    agent_pos = new_game_state["self"][-1]
    if has_agent_moved(old_game_state, new_game_state) and len(self.memory.shortest_paths_to_coin) > 0:
        for path in self.memory.shortest_paths_to_coin:
            self.logger.info(f"Compared path: {path} with position {agent_pos}")
            if agent_pos == path[1]:
                return True
    return False


def not_killed_by_own_bomb(events):
    return e.BOMB_EXPLODED in events and not e.KILLED_SELF in events


def unallowed_bomb(self_action, old_game_state):
    return self_action == "BOMB" and old_game_state["self"][2] == False


def has_agent_moved(old_game_state, new_game_state):
    return old_game_state["self"][-1] != new_game_state["self"][-1]
