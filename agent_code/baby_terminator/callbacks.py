import os
import pickle 
import random
import gc

import numpy as np
import gzip 

from .model import QNetwork, FullyConnectedQNetwork
import torch
import torch.optim as optim

from .memory import ReplayMemory

import math

from .utils import ACTIONS, device, DIRECTIONS, is_action_valid
from .path_finding import astar


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
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001, weight_decay=1e-5)
            self.memory = ReplayMemory(100000)

            weights = np.random.rand(len(ACTIONS))
            self.model = weights / weights.sum()
        else:
            self.logger.info("Using existing model to generate new generation")
            # with open("my-saved-model.pt", "rb") as file:
            #     self.policy_net, self.target_net, self.optimizer, self.memory = pickle.load(file)
            gc.disable()
            with gzip.open('my-saved-model.pkl.gz', 'rb') as f:
                self.policy_net, self.target_net, self.optimizer, self.memory = pickle.load(f)
            gc.enable()
    else:
        self.logger.info("Loading model from saved state, no training")
        # with open("my-saved-model.pt", "rb") as file:
        #     self.policy_net, self.target_net, self.optimizer, self.memory = pickle.load(file)
        gc.disable()
        with gzip.open('my-saved-model.pkl.gz', 'rb') as f:
            self.policy_net, self.target_net, self.optimizer, self.memory = pickle.load(f)
        gc.enable()


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    self.logger.info(50*"----")
    # Exploration vs exploitation
    # self.logger.info(game_state)
    if self.train:
        # Use epsilon greedy strategy to determine whether to exploit or explore
        EPS_START = 0.9
        EPS_END = 0.1
        EPS_DECAY = 300
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.memory.steps_done / EPS_DECAY)
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
            while not is_action_valid(self, game_state, action):
                self.logger.info(f"{action} is invalid... choosing again")
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
    
    field = game_state['field'].astype(np.float32)
    field[field == 1.] = 11.

    # The RGB map should contain:  
    # w = wall, e = empty, c = crate, a = agent, oa = other agents, b = bombs, co = coins
    # Define a mapping for the values: red, black and brown
    value_to_color = {-1: [255, 0, 0], 0: [0, 0, 0], 1: [210, 105, 30]}

    # Get the height and width of the original array
    height, width = game_state["field"].shape

    # Create a new array with 3 channels and the specified shape
    rgb_map = np.zeros((3, height, width), dtype=np.uint8)

    # Iterate through the original array and fill the new array with corresponding colors
    for i in range(height):
        for j in range(width):
            rgb_map[:, i, j] = value_to_color[game_state["field"][i, j]]

    agent_x, agent_y = game_state["self"][-1]
    # Add a agent at the specified coordinates with the color [1, 0, 255] = blue
    rgb_map[:, agent_x, agent_y] = [1, 0, 255]

    # Add other agents at the specified coordinates with the color [249, 0, 249] = pink
    for _, _, _, (x, y) in game_state['others']:
        rgb_map[:, x, y] = [249, 0, 249]

    # Add coins with the color [255, 255, 0] = yellow
    for (x, y) in game_state['coins']:
        rgb_map[:, x, y] = [255, 255, 0]

    # Add not exploded bombs with the color [0, 255, 0] = green and exploding bombs with explosion zone [0, 255, 222] = türkis
    for ((x, y), t) in game_state['bombs']:
        if t > 0:
            rgb_map[:, x, y] = [0, 255, 0]
        else:
            for dx in range(-3, 4):
                for dy in range(-3, 4): 
                    if inPlayArea(field, x+dx, y+dy) and (dx*dy == 0 and field[x+dx, y+dy] != 9):
                        rgb_map[:, x+dx, y+dy] = [0, 255, 222]

    # # Assuming new_array is the RGB image
    # import matplotlib.pyplot as plt
    # plt.imshow(rgb_map.transpose(1, 2, 0))  # Transpose to (height, width, channels) for display
    # plt.axis('off')  # Turn off axis labels
    # plt.show()

    # Convert the NumPy array to a torch tensor
    features_tensor = torch.from_numpy(rgb_map).float()

    # TODO: Maybe use min_max_scale?
    # Resize the tensor to have a shape of [3, 17, 17]
    features_tensor = features_tensor.view(3, 17, 17)

    return features_tensor


def channelwise_normalize_data(data):
    """
    Normalizes the data using Z-score normalization to not rely on batchnorm
    Args:
    - data (numpy.ndarray or torch.Tensor): Input data to be normalized.
    Returns:
    - normalized_data (numpy.ndarray or torch.Tensor): Normalized data.
    """
    mean = data.mean(dim=(1, 2), keepdim=True)
    std = data.std(dim=(1, 2), keepdim=True)
    
    normalized_data = (data - mean) / (std + 1e-7)  # Adding a small value to prevent division by zero
    
    return normalized_data


def min_max_scale(data):
    """
    Normalize the data by using a min max scaler strategie
    """
    minimum = torch.min(data)
    maximum = torch.max(data)

    return (data - minimum) / (maximum - minimum)

def inPlayArea(field, x, y):
    return (1 <= y < field.shape[0] - 1) and (1 <= x < field.shape[1] - 1)
