from collections import namedtuple, deque

import pickle
import gzip
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
UNALLOWED_BOMB = "UNALLOWED_BOMB"
DISTANCE_FROM_BOMB_INCREASED = "DISTANCE_FROM_BOMB_INCREASED"
DISTANCE_FROM_BOMB_DECREASED = "DISTANCE_FROM_BOMB_DECREASED"
DISTANCE_TO_COIN_INCREASED = "DISTANCE_TO_COIN_INCREASED"
DISTANCE_TO_COIN_DECREASED = "DISTANCE_TO_COIN_DECREASED"
APPROACHED_ENEMY = "APPROACHED_ENEMY"
DISAPPROACHED_ENEMY = "DISAPPROACHED_ENEMY"
LEFT_POTENTIAL_EXPLOSION_ZONE = "LEFT_POTENTIAL_EXPLOSION_ZONE"
ENTERED_POTENTIAL_EXPLOSION_ZONE = "ENTERED_POTENTIAL_EXPLOSION_ZONE"
IN_SAFE_ZONE = "IN_SAFE_ZONE"
AGENT_CORNERED = "AGENT_CORNERED"

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
    self.q_value = None
    self.loss = None



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
    custom_events = custom_game_events(self, old_game_state, new_game_state, events, self_action)
    events.extend(custom_events)
    self.logger.info("Game events occurred")
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    if old_game_state is None:
        self.logger.info("State is none in game events occurred")
        return
    # get the input for the CNN
    state = state_to_features(self, old_game_state)
    if state is not None:
        action = torch.tensor([ACTIONS.index(self_action)], device=device)
        reward = reward_from_events(self, events)
        if new_game_state is None:
            next_state = None
        else:
            next_state = state_to_features(self, new_game_state)
        reward = torch.tensor(reward, device=device)
        # push the state to the memory in order to be able to learn from it 
        self.memory.push(state, action, next_state, reward)
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
    custom_events = custom_game_events(self, None, last_game_state, events, last_action)
    events.extend(custom_events)
    state = state_to_features(self, last_game_state)
    action = torch.tensor([ACTIONS.index(last_action)], device=device)
    reward = reward_from_events(self, events)
    reward = torch.tensor(reward, device=device)
    self.memory.push(state, action, None, reward)

    # # synch the target network with the policy network 
    # self.target_net.load_state_dict(self.policy_net.state_dict())
    # self.target_net.eval()

    # Store the model

    with gzip.open('my-saved-model.pkl.gz', 'wb') as f:
        pickle.dump([self.policy_net, self.target_net, self.optimizer, self.memory], f)
    # with open("my-saved-model.pt", "wb") as file:
    #     pickle.dump([self.policy_net, self.target_net, self.optimizer, self.memory], file)

    # Add Q value to memory
    self.memory.q_value_after_episode.append(self.q_value)
    # Add loss to memory
    self.memory.loss_after_episode.append(self.loss)



def custom_game_events(self, old_game_state, new_game_state, events, self_action):
    custom_events = []
    valid_move = e.INVALID_ACTION in events
    in_old_explosion_zone = False
    in_new_explosion_zone = False
    # init with high value so if no bomb was in old state but one is discovered in new one, there is no penalty since 0 would be < distance to bomb
    old_distance_to_bomb = 1000
    new_distance_to_bomb = 1000
    # init with high value as safe space is set to ~3
    closest_bomb = 1000
    safe_distance = 2
    # init with high value so if no coin was in old state but one is discovered in new one, there is no penalty since 0 would be < distance to coin
    old_distance_to_coin = 1000
    new_distance_to_coin = 1000

    # if new is none something went wrong
    if new_game_state is None:
        return custom_events

    # append only the events that can also be calculated for the lats step
    if e.BOMB_EXPLODED in events and not e.KILLED_SELF in events:
        custom_events.append(NOT_KILLED_BY_OWN_BOMB)


    # if old is none --> Last round occured
    # append all events that can be calculated in the steps before the last one
    if old_game_state is not None:

        if self_action == "BOMB" and old_game_state["self"][2] == False:
            custom_events.append(UNALLOWED_BOMB)

        old_agent_pos = old_game_state["self"][-1]
        new_agent_pos = new_game_state["self"][-1]
        agent_moved = old_agent_pos != new_agent_pos

        if old_game_state["coins"]:
            old_distance_to_coin = min([abs(coin[0] - old_agent_pos[0]) + abs(coin[1] - old_agent_pos[1]) for coin in old_game_state["coins"]])
        if new_game_state["coins"]:
            new_distance_to_coin = min([abs(coin[0] - new_agent_pos[0]) + abs(coin[1] - new_agent_pos[1]) for coin in new_game_state["coins"]])

        if new_distance_to_coin < old_distance_to_coin and agent_moved and valid_move:
            self.logger.info(f"COIN DISTANCE DECREASED: {new_distance_to_coin} < {old_distance_to_coin}")
            custom_events.append("DISTANCE_TO_COIN_DECREASED")
        elif new_distance_to_coin > old_distance_to_coin and agent_moved and valid_move:
            self.logger.info(f"COIN DISTANCE INCREASED: {new_distance_to_coin} > {old_distance_to_coin}")
            custom_events.append("DISTANCE_TO_COIN_INCREASED")



        def explosion_zones(bomb_pos, explosion_radius=3):
            """Returns a list of coordinates that will be affected by the bomb's explosion."""
            x, y = bomb_pos
            zones = []

            # Add tiles for each direction until the explosion radius is reached
            for i in range(0, explosion_radius + 1):
                zones.extend([(x+i, y), (x-i, y), (x, y+i), (x, y-i)])
                
            return zones

        # check if there are bombs on the filed, if not skip calculations
        if old_game_state["bombs"]:
            old_distance_to_bomb = min([abs(bomb[0][0] - old_agent_pos[0]) + abs(bomb[0][1] - old_agent_pos[1]) for bomb in old_game_state["bombs"]])
            in_old_explosion_zone = any([old_agent_pos in explosion_zones(bomb_pos) for bomb_pos, _ in old_game_state["bombs"]])

        if new_game_state["bombs"]:
            new_distance_to_bomb = min([abs(bomb[0][0] - new_agent_pos[0]) + abs(bomb[0][1] - new_agent_pos[1]) for bomb in new_game_state["bombs"]])
            in_new_explosion_zone = any([new_agent_pos in explosion_zones(bomb_pos) for bomb_pos, _ in new_game_state["bombs"]])
            # preemptively calculate the closest bomb 
            closest_bomb = min([abs(bomb_pos[0] - new_agent_pos[0]) + abs(bomb_pos[1] - new_agent_pos[1]) for bomb_pos, _ in new_game_state["bombs"]], default=safe_distance)

        # entering does not imply a move since an enemy can plant the bomb nearby 
        if not in_old_explosion_zone and in_new_explosion_zone:
            self.logger.info(f"EXPLOSION ZONE ENTERED: {in_new_explosion_zone} < {in_old_explosion_zone}")
            custom_events.append("ENTERED_POTENTIAL_EXPLOSION_ZONE")
        elif in_old_explosion_zone and not in_new_explosion_zone and agent_moved and valid_move:
            self.logger.info(f"EXPLOSION ZONE LEFT: {in_new_explosion_zone} < {in_old_explosion_zone}")
            custom_events.append("LEFT_POTENTIAL_EXPLOSION_ZONE")

        # agent moved not neccessary since enemy can plant bomb nearby and a wait does decrease distance in this case
        if new_distance_to_bomb > old_distance_to_bomb:
            self.logger.info(f"BOMB DISTANCE INCREASED: {new_distance_to_bomb} > {old_distance_to_bomb}")
            custom_events.append("DISTANCE_FROM_BOMB_INCREASED")
        elif new_distance_to_bomb < old_distance_to_bomb and agent_moved and valid_move:
            self.logger.info(f"BOMB DISTANCE DECREASED: {new_distance_to_bomb} < {old_distance_to_bomb}")
            custom_events.append("DISTANCE_FROM_BOMB_DECREASED")


        old_distance_to_enemy = min([abs(enemy[-1][0] - old_agent_pos[0]) + abs(enemy[-1][1] - old_agent_pos[1]) for enemy in old_game_state["others"]])
        new_distance_to_enemy = min([abs(enemy[-1][0] - new_agent_pos[0]) + abs(enemy[-1][1] - new_agent_pos[1]) for enemy in new_game_state["others"]])
        # a wait might decrease the distance if the enemy approaches
        if new_distance_to_enemy < old_distance_to_enemy:
            self.logger.info(f"ENEMY DISTANCE DECREASED: {new_distance_to_enemy} < {old_distance_to_enemy}")
            custom_events.append("APPROACHED_ENEMY")
        elif new_distance_to_enemy > old_distance_to_enemy and agent_moved:
            self.logger.info(f"ENEMY DISTANCE DECREASED: {new_distance_to_enemy} < {old_distance_to_enemy}")
            custom_events.append("DISAPPROACHED_ENEMY")


        # Helper function to check if a position is blocked by walls or crates
        def is_blocked(position, game_state):
            x,y  = position
            return game_state['field'][x, y] == -1 or game_state['field'][x, y] == 1

        # Agent Cornered
        adjacent_positions = [(new_agent_pos[0] + dx, new_agent_pos[1] + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        blocked_directions = sum([is_blocked(pos, new_game_state) for pos in adjacent_positions])
        if blocked_directions >= 3:
            self.logger.info(f"Agent is blocked by {blocked_directions} tiles")
            custom_events.append("AGENT_CORNERED")


        # Safe Zone: No bombs or enemies nearby
        closest_enemy = min([abs(enemy[-1][0] - new_agent_pos[0]) + abs(enemy[-1][1] - new_agent_pos[1]) for enemy in new_game_state["others"]], default=safe_distance)
        if closest_bomb >= safe_distance and closest_enemy >= safe_distance:
            self.logger.info(f"IN SAFE ZONE {closest_bomb} > {safe_distance} and {closest_enemy} > {safe_distance}")
            custom_events.append("IN_SAFE_ZONE")

    return custom_events




def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.KILLED_OPPONENT: 75,
        e.INVALID_ACTION: -7.5,
        e.CRATE_DESTROYED: 15,
        e.COIN_FOUND: 15,
        e.COIN_COLLECTED: 20,
        e.KILLED_SELF: -25,
        e.GOT_KILLED: -25,
        e.MOVED_LEFT: 1,
        e.MOVED_RIGHT: 1,
        e.MOVED_UP: 1,
        e.MOVED_DOWN: 1,
        # waited penalty has to be bigger than safe zone reward
        e.WAITED: -5,
        e.BOMB_DROPPED: 1,
        e.BOMB_EXPLODED: 0,
        e.SURVIVED_ROUND: 50,
        e.OPPONENT_ELIMINATED: 10,
        NOT_KILLED_BY_OWN_BOMB: 35,
        # additional penalty when laying 2 bombs in a row
        UNALLOWED_BOMB: -20,
        DISTANCE_TO_COIN_DECREASED: 7.5,
        DISTANCE_TO_COIN_INCREASED: -5,
        DISTANCE_FROM_BOMB_INCREASED: 7.5,
        DISTANCE_FROM_BOMB_DECREASED: -5,
        APPROACHED_ENEMY: 5,
        DISAPPROACHED_ENEMY: -3,
        LEFT_POTENTIAL_EXPLOSION_ZONE: 7.5,
        ENTERED_POTENTIAL_EXPLOSION_ZONE: -5,
        IN_SAFE_ZONE: 2,
        AGENT_CORNERED: -10,

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
    UPDATE_FREQUENCY = 200
    if len(self.memory) < BATCH_SIZE:
        # if the memory does not contain enough information (< BATCH_SIZE) than do not learn
        return
    transitions = self.memory.sample(BATCH_SIZE -1 )
    # todo activate online learning approach --> you will need a larger memory for that > 1000000
    # "online learning"
    transitions.append(self.memory.memory[-1])
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None]).float()

    state_batch = torch.stack(batch.state).float()
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)

    # self.logger.info(f"Action-batch: {action_batch} | reward-batch: {reward_batch}")
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

    # Add Q value to object
    self.q_value = expected_state_action_values
    # Add loss to object
    self.loss = loss

    # back propagation
    self.optimizer.zero_grad()
    loss.backward()
    for param in self.policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    self.optimizer.step()

    # update the target net each C steps to be in synch with the policy net 
    if len(self.memory) % UPDATE_FREQUENCY == 0:
        self.logger.info("Update target network")
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

