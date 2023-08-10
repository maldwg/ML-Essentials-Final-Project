import os
import pickle
import random

import numpy as np
import gzip 

from .model import QNetwork, FullyConnectedQNetwork
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
    self.steps_done = 0
    if self.train:
        self.logger.info("Training mode selected")
        if not os.path.isfile("my-saved-model.pkl.gz"):
            self.logger.info("Setting up model from scratch.")
            # init policy and target network 
            self.policy_net = QNetwork(17, 17, 6).to(device)
            self.target_net = QNetwork(17, 17, 6).to(device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            self.optimizer = optim.RMSprop(self.policy_net.parameters())
            self.memory = ReplayMemory(1000)

            weights = np.random.rand(len(ACTIONS))
            self.model = weights / weights.sum()
        else:
            self.logger.info("Using existing model to generate new generation")
            # with open("my-saved-model.pt", "rb") as file:
            #     self.policy_net, self.target_net, self.optimizer, self.memory = pickle.load(file)
            with gzip.open('my-saved-model.pkl.gz', 'rb') as f:
                self.policy_net, self.target_net, self.optimizer, self.memory = pickle.load(f)
    else:
        self.logger.info("Loading model from saved state, no training")
        # with open("my-saved-model.pt", "rb") as file:
        #     self.policy_net, self.target_net, self.optimizer, self.memory = pickle.load(file)

        with gzip.open('my-saved-model.pkl.gz', 'rb') as f:
            self.policy_net, self.target_net, self.optimizer, self.memory = pickle.load(f)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # Exploration vs exploitation
    # self.logger.info(game_state)
    if self.train:
        # Use epsilon greedy strategy to determine whether to exploit or explore
        EPS_START = 0.5
        EPS_END = 0.05
        EPS_DECAY = 250
        EPS_MIN = 0.0
        sample = random.random()
        # let the exploration decay but not below 15 %
        eps_threshold = max(EPS_MIN, EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.memory.steps_done / EPS_DECAY))
        self.memory.steps_done += 1

        if sample > eps_threshold:
            self.logger.info("Exploitation")
            with torch.no_grad():
                state_features = state_to_features(self, game_state)
                state_features = state_features.unsqueeze(0).to(device)
                # self.logger.info("Game state transformed")
                # self.logger.info(state_features)
                # Pass features through policy network
                q_values = self.policy_net(state_features)
                self.logger.info(f"Q-values: {q_values} | Max Value chosen: {q_values.max(1)[1]} | Chosen view: {q_values.max(1)[1].view(1,1)} | Item: {q_values.max(1)[1].view(1,1).item()}")
                action = q_values.max(1)[1].view(1, 1)
                self.logger.info(f"Chose {ACTIONS[action.item()]} as best value ")
                return ACTIONS[action.item()]
        else:
            self.logger.info("Exploration")
            action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
            self.logger.info(f"Choose random action {action}")
            return action

    else:
        # exploit only in test mode
        self.logger.info("Exploitation")
        with torch.no_grad():
            state_features = state_to_features(self, game_state)
            state_features = state_features.unsqueeze(0).to(device)
            # Pass features through policy network           
            q_values = self.policy_net(state_features)
            self.logger.info(f"Q-values: {q_values} | Max Value chosen: {q_values.max(1)[1]} | Chosen view: {q_values.max(1)[1].view(1,1)} | Item: {q_values.max(1)[1].view(1,1).item()}")
            action = q_values.max(1)[1].view(1, 1)
            self.logger.info(f"Chose {ACTIONS[action.item()]} as best value ")
            return ACTIONS[action.item()]

def state_to_features(self, game_state: dict) -> np.array:
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
    
    field = game_state['field']
    field = np.transpose(field)
    field[field == 1.] = 11.
    field[field == 0.] = 10.
    field[field == -1.] = 9.

    agent_field = np.copy(field)
    x, y = game_state['self'][-1]
    agent_field[y, x] = 8.
    for _, _, _, (x, y) in game_state['others']:
        agent_field[y, x] = 7.

    explosion_map = np.transpose(game_state['explosion_map'])
    bomb_map = np.copy(field)
    bomb_map[explosion_map != 0] = explosion_map[explosion_map != 0]

    for ((x, y), t) in game_state['bombs']:
        bomb_map[y, x] = 20. + t

    coin_map = np.copy(field)
    for (x, y) in game_state['coins']:
        coin_map[y, x] = 6.

    # Safety Map
    safety_map = np.copy(field)
    for ((x, y), t) in game_state['bombs']:
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if (0 <= y+dy < safety_map.shape[0]) and (0 <= x+dx < safety_map.shape[1]):
                    safety_map[y+dy, x+dx] = 5

    # Threat Map
    threat_map = np.copy(field)
    for ((x, y), t) in game_state['bombs']:
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if (0 <= y+dy < threat_map.shape[0]) and (0 <= x+dx < threat_map.shape[1]):
                    threat_map[y+dy, x+dx] = t

    # Distance Map to next coin
    distance_map = np.copy(field)
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            # 30 is the maximum distance on the board
            min_distance = 30
            for (x, y) in game_state['coins']:
                min_distance = min(min_distance, abs(i-y) + abs(j-x))

            # update only free tiles with this info, otherwise no point
            if distance_map[i, j] == 10:
                distance_map[i, j] = min_distance

    # Dead Ends
    dead_ends = np.copy(field)
    for x in range(1, field.shape[0]-1):
        for y in range(1, field.shape[1]-1):
            # if cell is empty
            if field[x, y] == 10:  
                free_neighs = sum([field[x-1, y] == 10, field[x+1, y] == 10, field[x, y-1] == 10, field[x, y+1] == 10])
                if free_neighs == 1:
                    dead_ends[x, y] = 1

    # Threat from Enemies (Heatmap style)
    threat_map_enemies = np.copy(field)
    for _, _, _, (ex, ey) in game_state['others']:
        for x in range(field.shape[0]):
            for y in range(field.shape[1]):
                threat_map_enemies[x, y] += 1 / (1 + abs(ex - x) + abs(ey - y))

    # Potential Safe Zones
    potential_safe_zones = np.copy(field)
    for ((x, y), t) in game_state['bombs']:

        if t < 3 and t > 0:
            for dx in range(-3,4):
                for dy in range(-3,4):
                    # to only include the horizontal and vertical line of potential movement
                    if dx * dy == 0:
                        if 0 <= x + dx < field.shape[0] and 0 <= y + dy < field.shape[1]:
                            # set to 0 if a tile is *not* safe
                            potential_safe_zones[y + dy, x + dx] = 0

    # Stack all these features into a multi-channel tensor
    stacked_features = np.stack([agent_field, coin_map, bomb_map, safety_map, threat_map, distance_map, dead_ends, threat_map_enemies, potential_safe_zones], axis=0)

    # Convert to PyTorch tensor
    features_tensor = torch.from_numpy(stacked_features).float()
    features_tensor = normalize_data(features_tensor)

    return features_tensor


def normalize_data(data):
    """
    Normalizes the data using Z-score normalization to not rely on batchnorm
    Args:
    - data (numpy.ndarray or torch.Tensor): Input data to be normalized.
    Returns:
    - normalized_data (numpy.ndarray or torch.Tensor): Normalized data.
    """
    mean = data.mean()
    std = data.std()
    
    normalized_data = (data - mean) / (std + 1e-7)  # Adding a small value to prevent division by zero
    
    return normalized_data