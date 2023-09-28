import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import gzip
import json
from evaluations_utils import create_directory_if_not_exists, round_to_nearest_multiple
import subprocess


# parten directory containing all model files
parent_directory = "agent_code/baby_terminator/evaluation/stride_kernel_padding/"
pattern = "my-saved-model.pkl.gz"
hyperparameters_pattern = "parameters.json"

# dir for evaluation plots
figure_evaluation_dir = "agent_code/baby_terminator/evaluation/stride_kernel_padding/"
create_directory_if_not_exists(figure_evaluation_dir)

model_result = subprocess.run(
    ["find", parent_directory, "-type", "f", "-name", pattern],
    capture_output=True,
    text=True,
)
if model_result.returncode == 0:
    paths = model_result.stdout.strip().split("\n")
else:
    print("Error occurred while searching for files.")

model_numbers = subprocess.run(
    ["find", parent_directory, "-type", "d"],
    capture_output=True,
    text=True,
)


if model_numbers.returncode == 0:
    model_numbers = model_numbers.stdout.strip().split("\n")[1:]
else:
    print("Error occurred while searching for files.")

numbers = []
for m_nr in model_numbers:
    numbers.append(m_nr.split("/")[-1])

model_numbers = numbers


# List to store Q values, loss values, and rewards
q_values = []
losses = []
rewards = []
scores = []
# number of scores to look at each model
x = 20

# Iterate over each path
for model_path in paths:
    with gzip.open(model_path, "rb") as f:
        _, _, _, memory = pickle.load(f)

        # Get q_value and calc mean
        q_value_after_episode = memory.q_value_after_episode
        mean_q_values_after_episode = []
        for batch_q_value in q_value_after_episode:
            if isinstance(batch_q_value, torch.Tensor):
                mean_q_value_after_episode = torch.mean(batch_q_value).item()
                mean_q_values_after_episode.append(mean_q_value_after_episode)

        q_values.append(mean_q_values_after_episode)

        # Get loss and calc mean
        loss_after_episode = memory.loss_after_episode
        mean_losses_after_episode = []
        for batch_loss in loss_after_episode:
            if isinstance(batch_loss, torch.Tensor):
                mean_loss_after_episode = torch.mean(batch_loss).item()
                mean_losses_after_episode.append(mean_loss_after_episode)

        losses.append(mean_losses_after_episode)

        # Get rewards
        rewards.append(memory.rewards_after_round)

        # get last x rounds of scores
        scores.append(memory.rewards_after_round[-x:])

# Plot Q values
plt.figure(figsize=(12, 6), dpi=100)
for idx, mean_q_values_after_episode in enumerate(q_values):
    rounded_max_x = round_to_nearest_multiple(len(mean_q_values_after_episode))
    tick_interval = int(rounded_max_x / 5)
    if tick_interval == 0:
        tick_interval = 1

    if rounded_max_x < len(mean_q_values_after_episode):
        stop_x_tick = rounded_max_x + tick_interval + 1
    else:
        stop_x_tick = rounded_max_x + 1

    epochs = range(1, len(mean_q_values_after_episode) + 1)
    label = model_numbers[idx]
    plt.plot(epochs, mean_q_values_after_episode, label=label)
    plt.xticks(np.arange(0, stop_x_tick, tick_interval))


plt.title("Average Q Values per Episode")
plt.xlabel("Episode")
plt.ylabel("Q Value")

plt.legend(loc="best")

plt.savefig(f"{figure_evaluation_dir}grid_search_q_values.png")
plt.clf()

# Plot loss values
plt.figure(figsize=(12, 6), dpi=100)
for idx, mean_losses_after_episode in enumerate(losses):
    rounded_max_x = round_to_nearest_multiple(len(mean_losses_after_episode))
    tick_interval = int(rounded_max_x / 5)
    if tick_interval == 0:
        tick_interval = 1

    if rounded_max_x < len(mean_losses_after_episode):
        stop_x_tick = rounded_max_x + tick_interval + 1
    else:
        stop_x_tick = rounded_max_x + 1

    epochs = range(1, len(mean_losses_after_episode) + 1)
    label = model_numbers[idx]
    plt.plot(epochs, mean_losses_after_episode, label=label)
    plt.xticks(np.arange(0, stop_x_tick, tick_interval))

plt.title("Average Loss per Episode")
plt.xlabel("Episode")
plt.ylabel("Loss")

plt.legend(loc="best")
plt.savefig(f"{figure_evaluation_dir}grid_search_loss.png")
plt.clf()

# Plot rewards
plt.figure(figsize=(12, 6), dpi=100)
for idx, rewards_after_round in enumerate(rewards):
    rounds = [x for x in range(1, len(rewards_after_round) + 1)]

    rounded_max_x = round_to_nearest_multiple(len(rewards_after_round))
    tick_interval = int(rounded_max_x / 5)
    if tick_interval == 0:
        tick_interval = 1

    if rounded_max_x < len(rewards_after_round):
        stop_x_tick = rounded_max_x + tick_interval + 1
    else:
        stop_x_tick = rounded_max_x + 1

    plt.plot(rounds, rewards_after_round, label=model_numbers[idx])
    plt.xticks(np.arange(0, stop_x_tick, tick_interval))

plt.title("Total Rewards per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")

plt.legend(loc="best")
plt.savefig(f"{figure_evaluation_dir}grid_search_rewards.png")
plt.clf()
