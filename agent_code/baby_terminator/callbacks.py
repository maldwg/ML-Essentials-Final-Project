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
    self.steps_done = 0
    if self.train:
        self.logger.info("Training mode selected")
        if not os.path.isfile("my-saved-model.pt"):
            self.logger.info("Setting up model from scratch.")
            # init policy and target network 
            self.policy_net = QNetwork(17, 17, 6).to(device)
            self.target_net = QNetwork(17, 17, 6).to(device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            self.optimizer = optim.RMSprop(self.policy_net.parameters())
            self.memory = ReplayMemory(1000000)

            weights = np.random.rand(len(ACTIONS))
            self.model = weights / weights.sum()
        else:
            self.logger.info("Using existing model to generate new generation")
            with open("my-saved-model.pt", "rb") as file:
                self.policy_net, self.target_net, self.optimizer, self.memory = pickle.load(file)
    else:
        self.logger.info("Loading model from saved state, no training")
        with open("my-saved-model.pt", "rb") as file:
            self.policy_net, self.target_net, self.optimizer, self.memory = pickle.load(file)




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
        EPS_START = 0.9
        EPS_END = 0.05
        EPS_DECAY = 200
        sample = random.random()
        # let the exploration decay 
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            self.logger.info("Exploitation")
            with torch.no_grad():
                state_features = state_to_features(game_state)
                state_features = state_features.unsqueeze(0).to(device)
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
    else:
        # exploit only in test mode
        self.logger.info("Exploitation")
        with torch.no_grad():
            state_features = state_to_features(game_state)
            state_features = state_features.unsqueeze(0).to(device)
            # Pass features through policy network
            self.logger.info(f"Action values: {self.policy_net(state_features)}")
            action = self.policy_net(state_features).max(1)[1].view(1, 1)
            self.logger.info(f"Chose {ACTIONS[action.item()]} as best value ")
            return ACTIONS[action.item()]

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
    
    # Get field data
    field = game_state['field']
    # Transpose the field, because state is transposed!
    field = np.transpose(field)

    # Create coin map
    # y,x because the filed is transposed
    coin_map = np.zeros_like(field)
    for (x, y) in game_state['coins']:
        coin_map[y, x] = 3

    # Create agent's position channel
    agent_position = np.zeros_like(field)
    x, y = game_state['self'][-1]
    agent_position[y, x] = 2

    # Create other agents' position channels
    other_agents_positions = np.zeros_like(field)
    for _, _, _, (x, y) in game_state['others']:
        other_agents_positions[y, x] = -2

    # Normalize explosion map
    explosion_map = game_state['explosion_map']
    explosion_map = np.transpose(explosion_map)

    # Stack all these features into a multi-channel tensor
    stacked_features = np.stack([field, agent_position, other_agents_positions, explosion_map, coin_map], axis=0)

    # Convert to PyTorch tensor
    features_tensor = torch.from_numpy(stacked_features).float()
    return features_tensor

