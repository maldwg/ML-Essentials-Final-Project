import torch
from collections import namedtuple, deque
import events as e
from . import additional_events as ad

DIRECTIONS = [(1, 0), (0, 1), (-1, 0 ), (0, -1)]
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
device = torch.device("cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# TODO: adaptive rewards for invalid actions =?
game_rewards = {
        e.KILLED_OPPONENT: 50,
        e.INVALID_ACTION: -5,
        e.CRATE_DESTROYED: 15,
        e.COIN_FOUND: 5,
        e.COIN_COLLECTED: 25,
        e.KILLED_SELF: -50,
        e.GOT_KILLED: -50,
        e.MOVED_LEFT: 0.5,
        e.MOVED_RIGHT: 0.5,
        e.MOVED_UP: 0.5,
        e.MOVED_DOWN: 0.5,
        # waited penalty has to be bigger than safe zone reward
        e.WAITED: -3.5,
        e.BOMB_DROPPED: 0.5,
        e.BOMB_EXPLODED: 0,
        e.SURVIVED_ROUND: 25,
        e.OPPONENT_ELIMINATED: -2,
        ad.NOT_KILLED_BY_OWN_BOMB: 15,
        # additional penalty when laying 2 bombs in a row
        ad.UNALLOWED_BOMB: -5,
        ad.DISTANCE_TO_COIN_DECREASED: 5,
        ad.DISTANCE_TO_COIN_INCREASED: -4,
        # to prevent the agent from running away from all bombs
        # ad.DISTANCE_FROM_BOMB_INCREASED: 2,
        # ad.DISTANCE_FROM_BOMB_DECREASED: -1,
        ad.APPROACHED_ENEMY: 5,
        ad.DISAPPROACHED_ENEMY: -4,
        ad.LEFT_POTENTIAL_EXPLOSION_ZONE: 10,
        ad.ENTERED_POTENTIAL_EXPLOSION_ZONE: -5,
        # probably too dense ad.IN_SAFE_ZONE: 2,
        ad.AGENT_CORNERED: -1,
        ad.BOMB_DROPPED_BEFORE_CRATE: 5,
}

# TODO scenario based rewards ? 
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
#         ad.DISTANCE_TO_COIN_DECREASED: 0,
#         ad.DISTANCE_TO_COIN_INCREASED: 0,
#         ad.APPROACHED_ENEMY: 5,
#         ad.DISAPPROACHED_ENEMY: -4,
#         ad.LEFT_POTENTIAL_EXPLOSION_ZONE: 10,
#         ad.ENTERED_POTENTIAL_EXPLOSION_ZONE: -5,
#         ad.AGENT_CORNERED: 0,
#         ad.BOMB_DROPPED_BEFORE_CRATE: 0,
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
#         ad.DISTANCE_TO_COIN_DECREASED: 5,
#         ad.DISTANCE_TO_COIN_INCREASED: -5,
#         ad.APPROACHED_ENEMY: 5,
#         ad.DISAPPROACHED_ENEMY: -4,
#         ad.LEFT_POTENTIAL_EXPLOSION_ZONE: 10,
#         ad.ENTERED_POTENTIAL_EXPLOSION_ZONE: -5,
#         ad.AGENT_CORNERED: 0,
#         ad.BOMB_DROPPED_BEFORE_CRATE: 0,
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