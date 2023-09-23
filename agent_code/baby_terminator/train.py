from collections import deque

import pickle
import gc
import gzip
from typing import List

import numpy as np
from agent_code.baby_terminator.custom_event_handling import custom_game_events

from .callbacks import state_to_features

import torch
from torch import nn
from .utils import *


# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.logger.info("Enter train mode")
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.q_value = None
    self.loss = None
    self.number_of_executed_episodes = 0


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
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
    custom_events = custom_game_events(
        self, old_game_state, new_game_state, events, self_action
    )
    events.extend(custom_events)
    self.logger.debug(
        f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}'
    )
    if old_game_state is None:
        return
    # get the input for the CNN
    state = state_to_features(self, old_game_state)
    if state is not None:
        action = torch.tensor([ACTIONS.index(self_action)], device=device)
        reward = reward_from_events(self, events)
        self.memory.rewards_of_round.append(reward)
        if new_game_state is None:
            next_state = None
        else:
            next_state = state_to_features(self, new_game_state)
        self.logger.info(f"overall reward of step {reward}")
        reward = torch.tensor(reward, device=device)
        # push the state to the memory in order to be able to learn from it
        self.memory.push(state, action, next_state, reward)

        # needs to be before optimize otherwise the events occured are not taken into account
        increment_event_counts(self, events)

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
    self.logger.debug(
        f'Encountered event(s) {", ".join(map(repr, events))} in final step'
    )
    custom_events = custom_game_events(self, None, last_game_state, events, last_action)
    events.extend(custom_events)

    state = state_to_features(self, last_game_state)
    action = torch.tensor([ACTIONS.index(last_action)], device=device)
    reward = reward_from_events(self, events)

    self.memory.rewards_of_round.append(reward)
    overall_reward = sum(self.memory.rewards_of_round)
    self.logger.info(f"Overall reward at end of round: {overall_reward}")
    self.memory.rewards_after_round.append(overall_reward)
    # reset memory for next round
    self.memory.rewards_of_round = []

    reward = torch.tensor(reward, device=device)
    self.memory.push(state, action, None, reward)
    self.memory.shortest_paths_out_of_explosion = []
    self.memory.shortest_paths_to_coin = []
    self.memory.shortest_paths_to_enemy = []
    self.memory.shortest_paths_to_crate = []
    self.memory.left_explosion_zone = False
    optimize_model(self)

    # increment episode count
    self.number_of_executed_episodes += 1

    # Add Q value to memory
    self.memory.q_value_after_episode.append(self.q_value)
    # Add loss to memory
    self.memory.loss_after_episode.append(self.loss)

    # Store the model
    if self.number_of_executed_episodes == last_game_state["number_rounds"]:
        gc.disable()
        with gzip.open("my-saved-model.pkl.gz", "wb") as f:
            pickle.dump(
                [self.policy_net, self.target_net, self.optimizer, self.memory],
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        gc.enable()

def reward_from_events(self, events: List[str]) -> int:
    """
    Compute the reward based on game events.

    This function computes the cumulative reward for the agent based on the game events.

    :param self: The agent instance.
    :param events: List of game events that occurred.

    :return: int: The total reward.
    """

    reward_sum = 0
    rewarded_events = []
    for event in events:
        if event in self.memory.game_rewards:
            reward_sum += self.memory.game_rewards[event]
            rewarded_events.append(event)
    self.logger.info(
        f"Awarded {reward_sum} for the {len(rewarded_events)} events {', '.join(rewarded_events)}"
    )
    return reward_sum


def optimize_model(self):
    """
    Optimize the agent's policy network using Q-Learning methods.

    This function is responsible for optimizing the agent's policy network by computing the loss using the Q-Learning method and performing backpropagation.

    :param self: The agent instance.

    :return: None
    """
    self.logger.info("Optimizing model")
    # Adapt the hyper parameters
    BATCH_SIZE, GAMMA = self.memory.train_params.values()

    if len(self.memory) < BATCH_SIZE:
        # if the memory does not contain enough information (< BATCH_SIZE) than do not learn
        return
    transitions = self.memory.sample(BATCH_SIZE)
    # "online learning" by always including the last step to ensure we learn from this experience
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.stack(
        [s for s in batch.next_state if s is not None]
    ).float()

    state_batch = torch.stack(batch.state).float()
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)

    # Compute Q value for all actions taken in the batch
    state_action_values = self.policy_net(state_batch).gather(1, action_batch)

    # compute the expected Q values
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(
            1
        )[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # compute loss
    loss = nn.functional.smooth_l1_loss(
        state_action_values, expected_state_action_values.unsqueeze(1)
    )
    self.logger.info(f"Loss of {loss}")

    # Add Q value to object
    self.q_value = expected_state_action_values
    # Add loss to object
    self.loss = loss

    # back propagation
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
    self.optimizer.step()

    # Check if it's time to update the target network
    if self.memory.steps_since_last_update >= self.memory.update_frequency:
        self.logger.info("Update target network")
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        # Reset the steps since last update
        self.memory.steps_since_last_update = 0
    else:
        # Increment the counter
        self.memory.steps_since_last_update += 1

    # Dynamically adjust UPDATE_FREQUENCY via exp function only after the network has been updated once
    if self.memory.steps_since_last_update == 0:
        self.memory.update_frequency = int(
            500 * np.exp(0.00001 * self.memory.steps_done)
        )
        # Ensure there's a maximum limit for UPDATE_FREQUENCY to prevent very infrequent updates
        self.memory.update_frequency = min(self.memory.update_frequency, 3500)
