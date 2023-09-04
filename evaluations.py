import pickle
from matplotlib.pylab import plt
import numpy as np
import torch
import gzip
import os

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

AGENT_NAME = "baby_terminator"

model_path = "agent_code/" + AGENT_NAME + "/my-saved-model.pkl.gz"
figure_evaluation_dir = "agent_code/" + AGENT_NAME + "/evaluation/"
create_directory_if_not_exists(figure_evaluation_dir)


with gzip.open(model_path, 'rb') as f:
    policy_net,_,_, memory = pickle.load(f)

print(f"Model has remembered {memory.steps_done} steps")

 
########################################################
# Plot Q values
########################################################
# get q_value and calc mean
q_value_after_episode = memory.q_value_after_episode
mean_q_values_after_episode = []
for batch_q_value in q_value_after_episode:
    # Check that None is not passed
    if isinstance(batch_q_value, torch.Tensor):
        mean_q_value_after_episode = torch.mean(batch_q_value).item()
        mean_q_values_after_episode.append(mean_q_value_after_episode)

# Generate a sequence of integers to represent the epoch numbers
epochs = range(1, len(mean_q_values_after_episode) + 1)
plt.plot(epochs, mean_q_values_after_episode, label='Mean Q Value after each episode')
 
# Add in a title and axes labels
plt.title('Q Value after Episodes')
plt.xlabel('Episodes')
plt.ylabel('Q Value')
 
# Set the tick locations
tick_distance = int(len(mean_q_values_after_episode) / 5)
plt.xticks(np.arange(0, len(mean_q_values_after_episode) + 1, tick_distance))
 
# Display the plot and safe
plt.legend(loc='best')
plt.savefig(f"{figure_evaluation_dir}{AGENT_NAME}_q_values.png")
plt.clf()


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

plt.title('Loss after Episodes')
plt.xlabel('Episodes')
plt.ylabel('Loss')
plt.plot(epochs, mean_losses_after_episode, label='Mean Loss after each episode')
# Set the tick locations
plt.xticks(np.arange(0, len(mean_q_values_after_episode) + 1, tick_distance))

# Display the plot
plt.legend(loc='best')
plt.savefig(f"{figure_evaluation_dir}{AGENT_NAME}_loss.png")
plt.clf()

########################################################
# Plot distribution of CNN weights 
########################################################
def plot_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad and "bias" not in name:
            plt.figure(figsize=(15, 4))
            plt.title(name)
            plt.hist(param.data.cpu().numpy().flatten(), bins=100)
            plt.savefig(f"{figure_evaluation_dir}{AGENT_NAME}_{name}_weights.png")
            plt.clf()

plot_weights(policy_net)

########################################################
# Plot rewards
########################################################
rewards = memory.rewards_after_round
rounds = [ x for x in range(1, len(rewards) + 1)]
plt.title('Overall reward after Round')
plt.xlabel('Round')
plt.ylabel('Reward')
print(f"Length of overall trained rounds: {len(rewards)}")
plt.plot(rounds, rewards, label='Reward of round')
# Set the tick locations
plt.xticks(np.arange(0, len(mean_q_values_after_episode) + 1, tick_distance))

# Display the plot
plt.legend(loc='best')
plt.savefig(f"{figure_evaluation_dir}{AGENT_NAME}_rewards.png")
plt.clf()