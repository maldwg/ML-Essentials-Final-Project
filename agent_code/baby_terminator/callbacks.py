import os
import pickle
import random

import numpy as np

from .model import QNetwork
import torch
import torch.optim as optim

from .memory import ReplayMemory

import math

from .utils import ACTIONS, device 


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        # init policy and target network 
        self.policy_net = QNetwork(17, 17, 6).to(device)
        self.target_net = QNetwork(17, 17, 6).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)


        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.policy_net, self.target_net, self.optimizer, self.memory = pickle.load(file)


steps_done = 0

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # Exploration vs exploitation
    global steps_done
    # Use epsilon greedy strategy to determine whether to exploit or explore
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    sample = random.random()
    # let the exploration decay 
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        self.logger.info("Exploitation")
        with torch.no_grad():
            state_features = state_to_features(game_state)
            state_features = torch.from_numpy(state_features).float().unsqueeze(0).to(device)
            self.logger.info("Game state transformed")
            self.logger.info(state_features)
            # Pass features through policy network
            self.logger.info(self.policy_net(state_features))
            action = self.policy_net(state_features).max(1)[1].view(1, 1)
            self.logger.info(f"Chose {ACTIONS[action.item()]} as best value ")
            return ACTIONS[action.item()]
    else:
        self.logger.info("Exploration")
        action = torch.tensor([[random.randrange(6)]], device=device, dtype=torch.long)
        action = ACTIONS[action.item()]
        self.logger.info(f"Choose random action {action}")
        return action

def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    if game_state is None:
        return None
    
    # append the channels of the board to get the input shape for the CNN

    channels = []

    channels.append(game_state['field'])

    bomb_map = np.zeros_like(game_state['field'])
    for bomb in game_state['bombs']:
        bomb_map[bomb[0]] = bomb[1] 
    channels.append(bomb_map)

    coin_map = np.zeros_like(game_state['field'])
    for coin in game_state['coins']:
        coin_map[coin] = 1
    channels.append(coin_map)

    channels.append(game_state['explosion_map'])

    stacked_channels = np.stack(channels)

    return stacked_channels
