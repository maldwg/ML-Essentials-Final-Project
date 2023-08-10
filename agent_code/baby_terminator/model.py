import torch
from torch import nn

class QNetwork(nn.Module):
    def __init__(self, h, w, outputs):
        super(QNetwork, self).__init__()
        self.kernel_size = 3
        self.stride = 1
        self.padding = 2

        self.conv1 = nn.Conv2d(6, 8, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        # self.bn1 = nn.BatchNorm2d(8)
        #  self.dropout1 = nn.Dropout2d(p=0.5)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        # self.dropout2 = nn.Dropout2d(p=0.5)
        # self.conv3 = nn.Conv2d(8, 8, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        # self.dropout3 = nn.Dropout2d(p=0.5)
        def conv2d_size_out(size, kernel_size=self.kernel_size, stride=self.stride):
            """
            calculate the Output height or width of a convolutional layer
            """
            return (self.padding * 2 + size - kernel_size + stride) // stride 

        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))
        linear_input_size = convw * convh * 8
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        #x = self.bn1(x)
        #  x = self.dropout1(x)
        x = nn.functional.relu(self.conv2(x))
        # x = self.dropout2(x)
        # x = nn.functional.relu(self.conv3(x))
        # x = self.dropout3(x)
        x = self.head(x.view(x.size(0), -1))
        # return nn.functional.softmax(x)
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
