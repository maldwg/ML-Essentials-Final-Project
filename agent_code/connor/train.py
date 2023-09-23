import pickle
import gc
import gzip
from typing import List

from agent_code.baby_terminator.custom_event_handling import custom_game_events

from .callbacks import state_to_features

import torch
from .utils import *

import random

def setup_training(self):
    """
    Initialize the agent for training mode.

    This function is called after the `setup` function in `callbacks.py`.

    :param self: The agent instance.
    """
    self.logger.info("Enter train mode")
    self.loss = None
    self.number_of_executed_episodes = 0



def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """
    Called once per step for intermediate rewards based on game events.

    This function is responsible for storing game events and rewards at each game step.

    :param self: The agent instance.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: List of game events that occurred.

    :return: None
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

    state = state_to_features(self, old_game_state).unsqueeze(0).to(device)
    if state is not None:
        action = torch.tensor([ACTIONS.index(self_action)], device=device)
        reward = reward_from_events(self, events)
        self.memory.rewards_of_round.append(reward)

        self.logger.info(f"overall reward of step {reward}")
        reward = torch.tensor(reward, device=device)

        probabilities = self.policy_net(state)
        action = ACTIONS.index(self_action)
        log_prob = torch.log(probabilities[0][action])
        self.memory.step_action_rewards.append((log_prob, reward))

        increment_event_counts(self, events)


def end_of_round(self, last_game_state, last_action, events):
    """
    Called at the end of each game or when the agent died for final rewards.

    This function is responsible for handling the end-of-round updates for the agent, including model optimization.

    :param self: The agent instance.
    :param last_game_state: The last state of the game.
    :param last_action: The last action taken.
    :param events: List of game events that occurred.

    :return: None
    """
    self.logger.debug(
        f'Encountered event(s) {", ".join(map(repr, events))} in final step'
    )
    custom_events = custom_game_events(self, None, last_game_state, events, last_action)
    events.extend(custom_events)

    state = state_to_features(self, last_game_state).unsqueeze(0).to(device)
    reward = reward_from_events(self, events)

    self.memory.rewards_of_round.append(reward)
    overall_reward = sum(self.memory.rewards_of_round)
    self.logger.info(f"Overall reward at end of round: {overall_reward}")
    self.memory.rewards_after_round.append(overall_reward)
    # reset memory for next round
    self.memory.rewards_of_round = []

    probabilities = self.policy_net(state)
    action = ACTIONS.index(last_action)
    log_prob = torch.log(probabilities[0][action])

    reward = torch.tensor(reward, device=device)
    self.memory.step_action_rewards.append((log_prob, reward))
    self.memory.episode_action_rewards.append(self.memory.step_action_rewards)

    self.memory.shortest_paths_out_of_explosion = []
    self.memory.shortest_paths_to_coin = []
    self.memory.shortest_paths_to_enemy = []
    self.memory.shortest_paths_to_crate = []
    self.memory.step_action_rewards = []
    self.memory.left_explosion_zone = False
    optimize_model(self)

    # increment episode count
    self.number_of_executed_episodes += 1

    self.memory.loss_after_episode.append(self.loss)

    increment_event_counts(self, events)

    # Store the model
    if self.number_of_executed_episodes == last_game_state["number_rounds"]:
        gc.disable()
        with gzip.open("my-saved-model.pkl.gz", "wb") as f:
            pickle.dump(
                [self.policy_net, self.optimizer, self.memory],
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        gc.enable()

def reward_from_events(self, events):
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
    Optimize the agent's policy network using Policy Gradient methods.

    This function is responsible for optimizing the agent's policy network by computing the loss using the Policy Gradient method and performing backpropagation.

    :param self: The agent instance.

    :return: None
    """
    GAMMA = 0.9
    BATCHSIZE = 32

    if len(self.memory.episode_action_rewards) < BATCHSIZE:
        return

    episodes = random.sample(self.memory.episode_action_rewards, BATCHSIZE)
    self.memory.episode_action_rewards = list(
        filter(lambda i: i not in episodes, self.memory.episode_action_rewards)
    )
    loss = torch.tensor([])
    # iterate over each episode in batch
    for episode in episodes:
        # Compute the returns G_t for each timestep in the episode
        episode_loss = torch.zeros(len(episode))
        # compute loss for each step by using discounted reward (G) and log_prob
        for idx, (log_prob, _) in enumerate(episode):
            G = 0
            pw = 0
            for _, r in episode[idx:]:
                G = G + GAMMA**pw * r
                pw += 1
            # append loss for step at right position
            episode_loss[idx] = -log_prob * G
        # append episode loss to batch loss tensor
        loss = torch.cat((loss, episode_loss), dim=0)

    loss = loss.mean()

    self.loss = loss

    # back propagation
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
    self.optimizer.step()
