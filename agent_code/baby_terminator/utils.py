import torch
from collections import namedtuple, deque
import events as e
from . import custom_events as c
import functools
import numpy as np

DIRECTIONS = [(1, 0), (0, 1), (-1, 0 ), (0, -1)]
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
device = torch.device("cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


game_rewards = {

        # long term goal
        e.SURVIVED_ROUND: 100,
        c.SCORE_REWARD: 40,
        c.PLACEMENT_REWARD: 200,

        # killing goals
        e.KILLED_OPPONENT: 75,
        e.OPPONENT_ELIMINATED: -2,        
        e.KILLED_SELF: -100,
        e.GOT_KILLED: -50,

        # correct actions
        e.INVALID_ACTION: -10,
        # additional penalty when laying 2 bombs in a row
        c.UNALLOWED_BOMB: -10,

        # coin goals
        e.COIN_FOUND: 10,
        e.COIN_COLLECTED: 25,
        c.MOVED_TOWARDS_COIN: 7.5,

        # crate goals
        # crate destroyed im verhältnis zu coin found ändern, ggf. mehr für coin found als crate destroyed
        # little bit smaller since it is delayed --> adds up with the bomb_before_crate signal
        e.CRATE_DESTROYED: 7.5,
        c.CRATE_IN_EXPLOSION_ZONE: 12.5,
        # TODO: add shortened path to a coin or enemy ?
        # TODO: clarify: not explicitily enough
        # points can be assigned dynamically (old_shortest_path - new_shortest_path) * reward
        # ad.PATH_SHORTENED_TO_OBJECTIVE: 10,

        # bomb related goals
        c.MOVED_TOWARDS_END_OF_EXPLOSION: 2.5,
        c.LEFT_POTENTIAL_EXPLOSION_ZONE: 10,
        c.ENTERED_POTENTIAL_EXPLOSION_ZONE: -5,
        c.ATTACKED_ENEMY: 20,

        # penalize default actions otherwise too many watis and random moves
        # e.MOVED_DOWN: -0.1,
        # e.MOVED_LEFT: -0.1,
        # e.MOVED_RIGHT: -0.1,
        # e.MOVED_UP: -0.1,
        
        e.WAITED: -10,
        
        # Only give points if enemy is attacked or crate is in explosion zone
        e.BOMB_DROPPED: -10,

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


# Helper function to check if a position is blocked by walls or crates
def is_blocked(position, game_state):
    x,y  = position
    return game_state['field'][x, y] == -1 or game_state['field'][x, y] == 1