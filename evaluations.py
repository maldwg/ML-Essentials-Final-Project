import pickle
from matplotlib.pylab import plt
import numpy as np
import torch
import gzip

AGENT_NAME = "baby_terminator"

model_path = "agent_code/" + AGENT_NAME + "/my-saved-model.pkl.gz"

with gzip.open(model_path, 'rb') as f:
    policy_net,_,_, memory = pickle.load(f)

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
 
# Add in a title and axes labels
plt.title('Q Value after Episodes')
plt.xlabel('Episodes')
plt.ylabel('Q Value')
 
# Set the tick locations
plt.xticks(np.arange(0, len(mean_q_values_after_episode) + 1, 10))
 
# Display the plot
plt.legend(loc='best')
plt.show()
plt.savefig(f"./{AGENT_NAME}-q-values.png")
plt.clf()

# print loss 

plt.title('Loss after Episodes')
plt.xlabel('Episodes')
plt.ylabel('Loss')
plt.plot(epochs, mean_losses_after_episode, label='Mean Loss after each episode')
# Set the tick locations
plt.xticks(np.arange(0, len(mean_q_values_after_episode) + 1, 2))

# Display the plot
plt.legend(loc='best')
plt.show()
plt.savefig(f"./{AGENT_NAME}-loss.png")
plt.clf()


print(f"Model has remembered {memory.steps_done} steps")


def plot_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad and "bias" not in name:
            plt.figure(figsize=(15, 4))
            plt.title(name)
            plt.hist(param.data.cpu().numpy().flatten(), bins=100)
            plt.show()
            plt.savefig(f"./{AGENT_NAME}-weights.png")
            plt.clf()

plot_weights(policy_net)

