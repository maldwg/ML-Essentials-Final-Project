import os
import pickle
import gc

import numpy as np
import gzip

from .model import PolicyGradientNetwork
import torch
import torch.optim as optim

from .memory import ReplayMemory

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
    
            self.policy_net = PolicyGradientNetwork(17, 17, 6).to(device)
            self.optimizer = optim.Adam(
                self.policy_net.parameters(), lr=0.0001, weight_decay=1e-5
            )
            self.memory = ReplayMemory()

            weights = np.random.rand(len(ACTIONS))
            self.model = weights / weights.sum()
        else:
            self.logger.info("Using existing model to generate new generation")
            gc.disable()
            with gzip.open("my-saved-model.pkl.gz", "rb") as f:
                self.policy_net, self.optimizer, self.memory = pickle.load(f)
            gc.enable()
    else:
        self.logger.info("Loading model from saved state, no training")
        gc.disable()
        with gzip.open("my-saved-model.pkl.gz", "rb") as f:
            self.policy_net, self.optimizer, self.memory = pickle.load(f)
        gc.enable()


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    if self.train:
        with torch.no_grad():
            state_features = state_to_features(self, game_state)
            state_features = state_features.unsqueeze(0).to(device)
            probabilities = self.policy_net(state_features)
            # sample action according to softmax probability distribution output
            action = torch.multinomial(probabilities, num_samples=1).item()
            self.logger.info(f"chose action {ACTIONS[action]}")
            self.memory.steps_done += 1

            return ACTIONS[action]

    else:
        with torch.no_grad():
            state_features = state_to_features(self, game_state)
            state_features = state_features.unsqueeze(0).to(device)
            # Pass features through policy network
            probabilities = self.policy_net(state_features)
            # probabilities[0] because tensor is encapsulated in a list
            _, max_index = torch.max(probabilities[0], dim=0)
            action = max_index.item()
            self.logger.info(f"chose action {ACTIONS[action]}")
            return ACTIONS[action]


def state_to_features(self, game_state: dict) -> np.array:
    """
    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    self.logger.debug(f"Transforming the gamestate to RGB like picture")
    if game_state is None:
        return None

    field = game_state["field"].astype(np.float32)
    field[field == 1.0] = 11.0

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
    for _, _, _, (x, y) in game_state["others"]:
        rgb_map[:, x, y] = [249, 0, 249]

    # Add coins with the color [255, 255, 0] = yellow
    for x, y in game_state["coins"]:
        rgb_map[:, x, y] = [255, 255, 0]

    # Add not exploded bombs with the color [0, 255, 0] = green and exploding bombs with explosion zone [0, 255, 222] = cyan
    for (x, y), t in game_state["bombs"]:
        if t > 0:
            rgb_map[:, x, y] = [0, 255, 0]
        else:
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if inPlayArea(field, x + dx, y + dy) and (
                        dx * dy == 0 and field[x + dx, y + dy] != 9
                    ):
                        rgb_map[:, x + dx, y + dy] = [0, 255, 222]

    # Convert the NumPy array to a torch tensor
    features_tensor = torch.from_numpy(rgb_map).float()

    # Resize the tensor to have a shape of [3, 17, 17]
    features_tensor = features_tensor.view(3, 17, 17)

    return features_tensor


def inPlayArea(field, x, y):
    return (1 <= y < field.shape[0] - 1) and (1 <= x < field.shape[1] - 1)
