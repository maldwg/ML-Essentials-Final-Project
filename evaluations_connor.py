import pickle
from matplotlib.pylab import plt
import numpy as np
import torch
import gzip

from evaluations_utils import create_directory_if_not_exists, round_to_nearest_multiple

AGENT_NAME = "connor"

model_path = "agent_code/" + AGENT_NAME + "/my-saved-model.pkl.gz"
figure_evaluation_dir = "agent_code/" + AGENT_NAME + "/evaluation/"
create_directory_if_not_exists(figure_evaluation_dir)


with gzip.open(model_path, "rb") as f:
    policy_net, _, memory = pickle.load(f)

print(f"Model has remembered {memory.steps_done} steps")
print(f"Memmory length: {len(memory)}")


########################################################
# Plot loss values
########################################################
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

epochs = range(1, len(mean_losses_after_episode) + 1)
plt.figure(figsize=(12, 6), dpi=100)
plt.plot(epochs, mean_losses_after_episode, label="Loss after each Episode")

plt.title("Loss after Episodes")
plt.xlabel("Episode")
plt.ylabel("Loss")

# Set the tick locations
plt.xticks(np.arange(0, stop_x_tick, tick_interval))

# Display the plot
plt.legend(loc="best")
plt.savefig(f"{figure_evaluation_dir}{AGENT_NAME}_loss.png")
plt.clf()


########################################################
# Plot distribution of CNN weights
########################################################
def plot_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad and "bias" not in name:
            plt.figure(figsize=(12, 6), dpi=100)
            plt.title(name)
            plt.hist(param.data.cpu().numpy().flatten(), bins=100)
            plt.savefig(f"{figure_evaluation_dir}{AGENT_NAME}_{name}_weights.png")
            plt.clf()


plot_weights(policy_net)

########################################################
# Plot rewards
########################################################
rewards = memory.rewards_after_round
rounds = [x for x in range(1, len(rewards) + 1)]

rounded_max_x = round_to_nearest_multiple(len(rewards))
tick_interval = int(rounded_max_x / 5)
if tick_interval == 0:
    tick_interval = 1

if rounded_max_x < len(rewards):
    stop_x_tick = rounded_max_x + tick_interval + 1
else:
    stop_x_tick = rounded_max_x + 1

plt.figure(figsize=(12, 6), dpi=100)
plt.title("Overall reward after Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
print(f"Length of overall trained rounds: {len(rewards)}")
plt.plot(rounds, rewards, label="Reward of round")
# Set the tick locations
plt.xticks(np.arange(0, stop_x_tick, tick_interval))

# Display the plot
plt.legend(loc="best")
plt.savefig(f"{figure_evaluation_dir}{AGENT_NAME}_rewards.png")
plt.clf()
