import torch
from torch import nn

class QNetwork(nn.Module):
    def __init__(self, h, w, outputs):
        super(QNetwork, self).__init__()
        self.kernel_size = 3
        self.stride = 1
        self.padding = 2

        self.conv1 = nn.Conv2d(9, 18, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        def conv2d_size_out(size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding):
            """
            calculate the Output height or width of a convolutional layer
            """
            return (padding * 2 + size - kernel_size + stride) // stride 

        convw = conv2d_size_out(w)
        convh = conv2d_size_out(h)
        linear_input_size = convw * convh * 18
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.head(x.view(x.size(0), -1))
        return x 


import torch.nn.functional as F

class FullyConnectedQNetwork(nn.Module):
    def __init__(self, input_size, outputs):
        super(FullyConnectedQNetwork, self).__init__()
        
        # Define the size of the hidden layers
        hidden_1 = 512
        hidden_2 = 256
        hidden_3 = 128
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, hidden_3)
        self.fc4 = nn.Linear(hidden_3, outputs)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x
