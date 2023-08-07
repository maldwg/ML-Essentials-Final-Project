import pickle
from matplotlib.pylab import plt
import numpy as np
import torch


AGENT_NAME = "baby_terminator"

model_path = "agent_code/" + AGENT_NAME + "/my-saved-model.pt"
with open(model_path, "rb") as file:
    _, _, _, memory = pickle.load(file)

# get q_value and calc mean
q_value_after_episode = memory.q_value_after_episode
mean_q_values_after_episode = []
for batch_q_value in q_value_after_episode:
    # Check that None is not passed
    if isinstance(batch_q_value, torch.Tensor):
        mean_q_value_after_episode = torch.mean(batch_q_value).item()
        mean_q_values_after_episode.append(mean_q_value_after_episode)

# get loss and calc mean
loss_after_episode = memory.loss_after_episode
mean_losses_after_episode = []
for batch_loss in loss_after_episode:
    # Check that None is not passed
    if isinstance(batch_loss, torch.Tensor):
        mean_loss_after_episode = torch.mean(batch_loss).item()
        mean_losses_after_episode.append(mean_loss_after_episode)


# Generate a sequence of integers to represent the epoch numbers
epochs = range(1, len(mean_q_values_after_episode) + 1)
 
# Plot and label the training and validation loss values
plt.plot(epochs, mean_q_values_after_episode, label='Mean Q Value after each episode')
plt.plot(epochs, mean_losses_after_episode, label='Mean Loss after each episode')
 
# Add in a title and axes labels
plt.title('Q Value and Loss after Episodes')
plt.xlabel('Episodes')
plt.ylabel('Loss and Q Value')
 
# Set the tick locations
plt.xticks(np.arange(0, len(mean_q_values_after_episode) + 1, 2))
 
# Display the plot
plt.legend(loc='best')
plt.show()