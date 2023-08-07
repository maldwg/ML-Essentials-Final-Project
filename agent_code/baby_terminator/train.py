from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features

from .model import QNetwork
from .memory import ReplayMemory

import torch
from torch import nn
from .utils import ACTIONS, device, Transition


# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
NOT_KILLED_BY_OWN_BOMB = "NOT_KILLED_BY_OWN_BOMB"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.logger.info("Enter train mode")
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    if e.BOMB_EXPLODED in events and not e.KILLED_SELF in events:
        events.append(NOT_KILLED_BY_OWN_BOMB)
    self.logger.info("Game events occurred")
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    if old_game_state is None:
        self.logger.info("State is none in game events occurred")
        return
    # get the input for the CNN
    state = state_to_features(old_game_state)
    if state is not None:
        action = torch.tensor([ACTIONS.index(self_action)], device=device)
        reward = reward_from_events(self, events)
        if new_game_state is None:
            next_state = None
        else:
            next_state = state_to_features(new_game_state)
        reward = torch.tensor(reward, device=device)
        # push the state to the memory in order to be able to learn from it 
        self.memory.push(torch.tensor(state), action, torch.tensor(next_state), reward)
        optimize_model(self)

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    state = state_to_features(last_game_state)
    action = torch.tensor([ACTIONS.index(last_action)], device=device)
    reward = reward_from_events(self, events)
    reward = torch.tensor(reward, device=device)
    self.memory.push(torch.tensor(state), action, None, reward)

    # # synch the target network with the policy network 
    # self.target_net.load_state_dict(self.policy_net.state_dict())
    # self.target_net.eval()

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump([self.policy_net, self.target_net, self.optimizer, self.memory], file)

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.KILLED_OPPONENT: 25,
        e.INVALID_ACTION: -10,
        e.CRATE_DESTROYED: 5,
        e.COIN_FOUND: 1,
        e.COIN_COLLECTED: 10,
        e.KILLED_SELF: -150,
        e.GOT_KILLED: -50,
        e.MOVED_LEFT: 0,
        e.MOVED_RIGHT: 0,
        e.MOVED_UP: 0,
        e.MOVED_DOWN: 0,
        e.WAITED: -1,
        e.BOMB_DROPPED: 0,
        e.BOMB_EXPLODED: 2,
        e.SURVIVED_ROUND: 5,
        e.OPPONENT_ELIMINATED: 5,
        NOT_KILLED_BY_OWN_BOMB: 300
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def optimize_model(self):
    """
    Method to perform the actual Q-Learning
    """
    self.logger.info("Optimizing model")
    # Adapt the hyper parameters
    BATCH_SIZE = 128
    GAMMA = 0.999
    UPDATE_FREQUENCY = 100
    if len(self.memory) < BATCH_SIZE:
        # if the memory does not contain enough information (< BATCH_SIZE) than do not learn
        return
    transitions = self.memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None]).float()

    state_batch = torch.stack(batch.state).float()
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)
    # Construct Q value for the current state
    state_action_values = self.policy_net(state_batch).gather(1, action_batch)

    # compute the expected Q values
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch 

    # compute loss
    loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    self.logger.info(f"Q value of: {expected_state_action_values}")
    self.logger.info(f"Loss of {loss}")
    # back propagation
    self.optimizer.zero_grad()
    loss.backward()
    for param in self.policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    self.optimizer.step()

    # update the target net each C steps to be in synch with the policy net 
    if len(self.memory) % UPDATE_FREQUENCY:
        self.logger.info("Update target network")
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

