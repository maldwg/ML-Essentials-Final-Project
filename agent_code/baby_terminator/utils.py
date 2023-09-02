import torch
from collections import namedtuple, deque
import events as e
from . import additional_events as ad
import functools
import numpy as np

DIRECTIONS = [(1, 0), (0, 1), (-1, 0 ), (0, -1)]
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
device = torch.device("cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


game_rewards = {

        # long term goal
        e.SURVIVED_ROUND: 75,
        ad.SCORE_REWARD: 25,
        ad.PLACEMENT_REWARD: 200,

        # killing goals
        e.KILLED_OPPONENT: 50,
        e.OPPONENT_ELIMINATED: -2,        
        e.KILLED_SELF: -50,
        e.GOT_KILLED: -50,

        # correct actions
        e.INVALID_ACTION: -5,
        # additional penalty when laying 2 bombs in a row
        ad.UNALLOWED_BOMB: -5,

        # coin goals
        e.COIN_FOUND: 5,
        e.COIN_COLLECTED: 25,
        ad.MOVED_TOWARDS_COIN: 5,

        # crate goals
        # crate destroyed im verhältnis zu coin found ändern, ggf. mehr für coin found als crate destroyed
        # little bit smaller since it is delayed --> adds up with the bomb_before_crate signal
        e.CRATE_DESTROYED: 15,
        ad.CRATE_IN_EXPLOSION_ZONE: 5,
        # TODO: add shortened path to a coin or enemy ?
        # TODO: clarify: not explicitily enough
        # points can be assigned dynamically (old_shortest_path - new_shortest_path) * reward
        # ad.PATH_SHORTENED_TO_OBJECTIVE: 10,

        # bomb related goals
        ad.MOVED_TOWARDS_END_OF_EXPLOSION: 5,
        ad.LEFT_POTENTIAL_EXPLOSION_ZONE: 10,
        ad.ENTERED_POTENTIAL_EXPLOSION_ZONE: -5,
        ad.ATTACKED_ENEMY: 10,

}

# for the empty field scenario
# game_rewards = {
#         e.KILLED_OPPONENT: 50,
#         e.INVALID_ACTION: -7.5,
#         e.CRATE_DESTROYED: 0,
#         e.COIN_FOUND: 0,
#         e.COIN_COLLECTED: 0,
#         e.KILLED_SELF: -50,
#         e.GOT_KILLED: -50,
#         e.MOVED_LEFT: 0.5,
#         e.MOVED_RIGHT: 0.5,
#         e.MOVED_UP: 0.5,
#         e.MOVED_DOWN: 0.5,
#         e.WAITED: -3.5,
#         e.BOMB_DROPPED: 0.5,
#         e.BOMB_EXPLODED: 0,
#         e.SURVIVED_ROUND: 25,
#         e.OPPONENT_ELIMINATED: -2,
#         ad.NOT_KILLED_BY_OWN_BOMB: 15,
#         # additional penalty when laying 2 bombs in a row
#         ad.UNALLOWED_BOMB: -5,
#         ad.MOVED_TOWARDS_COIN: 0,
#         ad.DISTANCE_TO_COIN_INCREASED: 0,
#         ad.APPROACHED_ENEMY: 5,
#         ad.DISAPPROACHED_ENEMY: -4,
#         ad.LEFT_POTENTIAL_EXPLOSION_ZONE: 10,
#         ad.ENTERED_POTENTIAL_EXPLOSION_ZONE: -5,
#         ad.AGENT_CORNERED: 0,
#         ad.CRATE_IN_EXPLOSION_ZONE: 0,
# }


# for the coin-heaven field scenario
# game_rewards = {
#         e.KILLED_OPPONENT: 50,
#         e.INVALID_ACTION: -7.5,
#         e.CRATE_DESTROYED: 0,
#         e.COIN_FOUND: 5,
#         e.COIN_COLLECTED: 10,
#         e.KILLED_SELF: -50,
#         e.GOT_KILLED: -50,
#         e.MOVED_LEFT: 0.5,
#         e.MOVED_RIGHT: 0.5,
#         e.MOVED_UP: 0.5,
#         e.MOVED_DOWN: 0.5,
#         e.WAITED: -3.5,
#         e.BOMB_DROPPED: 0.5,
#         e.BOMB_EXPLODED: 0,
#         e.SURVIVED_ROUND: 25,
#         e.OPPONENT_ELIMINATED: -2,
#         ad.NOT_KILLED_BY_OWN_BOMB: 15,
#         # additional penalty when laying 2 bombs in a row
#         ad.UNALLOWED_BOMB: -5,
#         ad.MOVED_TOWARDS_COIN: 5,
#         ad.DISTANCE_TO_COIN_INCREASED: -5,
#         ad.APPROACHED_ENEMY: 5,
#         ad.DISAPPROACHED_ENEMY: -4,
#         ad.LEFT_POTENTIAL_EXPLOSION_ZONE: 10,
#         ad.ENTERED_POTENTIAL_EXPLOSION_ZONE: -5,
#         ad.AGENT_CORNERED: 0,
#         ad.CRATE_IN_EXPLOSION_ZONE: 0,
# }

# BAD!!!1!
# game_rewards = {
#     e.INVALID_ACTION: -100,
#     e.CRATE_DESTROYED: 100,
#     e.COIN_COLLECTED: 400,
#     e.WAITED: -15,
#     e.BOMB_DROPPED: -20,
#     e.KILLED_SELF: -500,
#     e.KILLED_OPPONENT: 400,
#     e.GOT_KILLED: -500,
#     e.SURVIVED_ROUND: 300,
#     e.MOVED_DOWN:-1,
#     e.MOVED_LEFT:-1,
#     e.MOVED_RIGHT:-1,
#     e.MOVED_UP:-1,
#     e.COIN_FOUND:-1,
#     e.BOMB_EXPLODED:-1
# }




def tiles_beneath_explosion(self, new_game_state, potential_explosions):
    neighbours = []
    for x,y in potential_explosions:
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


def reshape_rewards(self):
    self.logger.info("Updating rewards...")
    # self.logger.info(f"old rewards: {self.memory.game_rewards}")
    # sum up all events so far seen
    event_counts = self.memory.rewarded_event_counts
    total_events = functools.reduce(lambda ac,k: ac+event_counts[k], event_counts, 0)
    # calculate fractions of event counts
    event_counts = {key: round(event_counts[key] / total_events, 3) for key in event_counts}
    average_event_fraction = np.mean(list(event_counts.values()))
    self.logger.info(f"Average occurrence: {average_event_fraction}")
    self.logger.info(f"fractioned event counts {event_counts}")
    for event in self.memory.game_rewards:
        bonus = abs(self.memory.game_rewards[event]) * (average_event_fraction - event_counts[event]) 
        # self.memory.game_rewards[event] += bonus
    self.logger.info(f"Updated rewards: { self.memory.game_rewards}")

def increment_event_counts(self, events):
    self.logger.info("Increment count of events in memory")
    for event in events:
        if event in self.memory.rewarded_event_counts:
            self.memory.rewarded_event_counts[event] += 1
    # self.logger.info(f"incremented events: {self.memory.rewarded_event_counts}")