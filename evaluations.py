import pickle
from matplotlib.pylab import plt
import numpy as np


AGENT_NAME = "baby_terminator"

model_path = "agent_code/" + AGENT_NAME + "/my-saved-model.pt"
with open(model_path, "rb") as file:
    _, _, _, memory = pickle.load(file)

print(memory)
print(memory.q_value_after_episode)
print(memory.loss_after_episode)


# # Load the training and validation loss dictionaries
# train_loss = load(open('train_loss.pkl', 'rb'))
# val_loss = load(open('val_loss.pkl', 'rb'))
 
# # Retrieve each dictionary's values
# train_values = train_loss.values()
# val_values = val_loss.values()
 
# # Generate a sequence of integers to represent the epoch numbers
# epochs = range(1, 21)
 
# # Plot and label the training and validation loss values
# plt.plot(epochs, train_values, label='Training Loss')
# plt.plot(epochs, val_values, label='Validation Loss')
 
# # Add in a title and axes labels
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
 
# # Set the tick locations
# plt.xticks(arange(0, 21, 2))
 
# # Display the plot
# plt.legend(loc='best')
# plt.show()