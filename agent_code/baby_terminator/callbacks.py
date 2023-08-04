import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']


def setup(self):
    np.random.seed()


def act(agent, game_state: dict):
    agent.logger.info('Pick action at random, but no bombs.')
    agent.logger.info(game_state)
    return np.random.choice(ACTIONS)
