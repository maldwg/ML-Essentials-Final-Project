import pickle
from matplotlib.pylab import plt
import numpy as np
import torch
import gzip

from evaluations_utils import create_directory_if_not_exists, round_to_nearest_multiple


import subprocess

# parten directory containing all model files
parent_directory = "agent_code/baby_terminator/checkpoints/350-coin-heaven-200-was-best/"
pattern = "my-saved-model.pkl.gz"

# dir for evaluation plots
figure_evaluation_dir = "agent_code/baby_terminator/checkpoints/350-coin-heaven-200-was-best/evaluation/"
create_directory_if_not_exists(figure_evaluation_dir)

result = subprocess.run(["find", parent_directory, "-type", "f", "-name", pattern], capture_output=True, text=True)
if result.returncode == 0:
    paths = result.stdout.strip().split('\n')
else:
    print("Error occurred while searching for files.")

# list of memories
memories = []
# iterate over each path
for model_path in paths:
    with gzip.open(model_path, 'rb') as f:
        _, _, _, memory = pickle.load(f)
        memories.append(memory)


########################################################
# Plot Q values
########################################################
for memory in memories:
    # get q_value and calc mean
    q_value_after_episode = memory.q_value_after_episode
    mean_q_values_after_episode = []
    for batch_q_value in q_value_after_episode:
        # Check that None is not passed
        if isinstance(batch_q_value, torch.Tensor):
            mean_q_value_after_episode = torch.mean(batch_q_value).item()
            mean_q_values_after_episode.append(mean_q_value_after_episode)

    rounded_max_x = round_to_nearest_multiple(len(mean_q_values_after_episode))
    tick_interval = int(rounded_max_x / 5)
    if tick_interval == 0:
        tick_interval = 1

    if rounded_max_x < len(mean_q_values_after_episode): 
        stop_x_tick = rounded_max_x + tick_interval + 1
    else:
        stop_x_tick = rounded_max_x + 1

    # Generate a sequence of integers to represent the epoch numbers
    epochs = range(1, len(mean_q_values_after_episode) + 1)
    # TODO: change label
    label = "Hyperparameters"
    plt.plot(epochs, mean_q_values_after_episode, label=label)
    
# Add in a title and axes labels
plt.title('Q Values after Episodes for top 10 Grid Search models')
plt.xlabel('Episodes')
plt.ylabel('Q Value')

# Set the tick locations
plt.xticks(np.arange(0, stop_x_tick, tick_interval))
 
# Display the plot and safe
plt.legend(loc='best')
plt.savefig(f"{figure_evaluation_dir}grid_search_top_10_q_values.png")
plt.clf()


########################################################
# Plot loss values
########################################################
for memory in memories:
    # get loss and calc mean
    loss_after_episode = memory.loss_after_episode
    mean_losses_after_episode = []
    for batch_loss in loss_after_episode:
        # Check that None is not passed
        if isinstance(batch_loss, torch.Tensor):
            mean_loss_after_episode = torch.mean(batch_loss).item()
            mean_losses_after_episode.append(mean_loss_after_episode)

    rounded_max_x = round_to_nearest_multiple(len(mean_losses_after_episode))
    tick_interval = int(rounded_max_x / 5)
    if tick_interval == 0:
        tick_interval = 1

    if rounded_max_x < len(mean_losses_after_episode): 
        stop_x_tick = rounded_max_x + tick_interval + 1
    else:
        stop_x_tick = rounded_max_x + 1

    # Generate a sequence of integers to represent the epoch numbers
    epochs = range(1, len(mean_losses_after_episode) + 1)

    # TODO: rename label
    label = "Hyperparameter"
    plt.plot(epochs, mean_losses_after_episode, label=label)
    # Set the tick locations
    plt.xticks(np.arange(0, stop_x_tick, tick_interval))

plt.title('Loss after Episodes for top 10 Grid Search models')
plt.xlabel('Episodes')
plt.ylabel('Loss')

# Display the plot
plt.legend(loc='best')
plt.savefig(f"{figure_evaluation_dir}grid_search_top_10_loss.png")
plt.clf()


########################################################
# Plot rewards
########################################################
for memory in memories:
    rewards = memory.rewards_after_round
    rounds = [ x for x in range(1, len(rewards) + 1)]

    rounded_max_x = round_to_nearest_multiple(len(rewards))
    tick_interval = int(rounded_max_x / 5)
    if tick_interval == 0:
        tick_interval = 1

    if rounded_max_x < len(mean_q_values_after_episode): 
        stop_x_tick = rounded_max_x + tick_interval + 1
    else:
        stop_x_tick = rounded_max_x + 1

    plt.plot(rounds, rewards, label='Reward of round')
    # Set the tick locations
    plt.xticks(np.arange(0, stop_x_tick, tick_interval))

plt.title('Overall reward after Round')
plt.xlabel('Round')
plt.ylabel('Reward')

# Display the plot
plt.legend(loc='best')
plt.savefig(f"{figure_evaluation_dir}grid_search_top_10_rewards.png")
plt.clf()