import torch
from collections import namedtuple, deque
import events as e

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
device = torch.device("cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

game_rewards = {
        e.KILLED_OPPONENT: 50,
        e.INVALID_ACTION: -12.5,
        e.CRATE_DESTROYED: 15,
        e.COIN_FOUND: 15,
        e.COIN_COLLECTED: 25,
        e.KILLED_SELF: -15,
        e.GOT_KILLED: -15,
        e.MOVED_LEFT: 0.5,
        e.MOVED_RIGHT: 0.5,
        e.MOVED_UP: 0.5,
        e.MOVED_DOWN: 0.5,
        # waited penalty has to be bigger than safe zone reward
        e.WAITED: -7.5,
        e.BOMB_DROPPED: 0.5,
        e.BOMB_EXPLODED: 0,
        e.SURVIVED_ROUND: 25,
        e.OPPONENT_ELIMINATED: 5,
        NOT_KILLED_BY_OWN_BOMB: 15,
        # additional penalty when laying 2 bombs in a row
        UNALLOWED_BOMB: -10,
        DISTANCE_TO_COIN_DECREASED: 5,
        DISTANCE_TO_COIN_INCREASED: -4,
        DISTANCE_FROM_BOMB_INCREASED: 5,
        DISTANCE_FROM_BOMB_DECREASED: -4,
        APPROACHED_ENEMY: 5,
        DISAPPROACHED_ENEMY: -4,
        LEFT_POTENTIAL_EXPLOSION_ZONE: 5,
        ENTERED_POTENTIAL_EXPLOSION_ZONE: -4,
        IN_SAFE_ZONE: 2,
        AGENT_CORNERED: -7.5,
}

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