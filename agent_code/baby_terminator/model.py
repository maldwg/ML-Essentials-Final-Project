import torch
from torch import nn

class QNetwork(nn.Module):
    def __init__(self, h, w, outputs):


        super(QNetwork, self).__init__()
        self.kernel_size = 3
        self.stride = 2
        self.padding = 0

        self.conv1 = nn.Conv2d(3, 16, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.dropout1 = nn.Dropout(0.3)
        # self.pooling1 = nn.MaxPool2d(self.kernel_size, self.stride)
        

        self.conv2 = nn.Conv2d(16, 32, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.dropout2 = nn.Dropout(0.3)
        # self.pooling2 = nn.MaxPool2d(self.kernel_size, self.stride)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.dropout3 = nn.Dropout(0.3)

        def conv2d_size_out(size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding):
            """
            calculate the Output height or width of a convolutional layer
            """
            return (padding * 2 + size - kernel_size + stride) // stride 

        def conv2d_size_out_with_pooling(size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding):
            """
            calculate the Output height or width of a convolutional layer
            """
            cnn_output =  (padding * 2 + size - kernel_size + stride) // stride 
            pooling_output = (cnn_output - kernel_size + stride ) // stride
            return pooling_output

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 64
        assert linear_input_size == 64, "convw or convh is not 1"

        # self.conv3 = nn.Conv2d(32, 64, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        # self.pooling3 = nn.MaxPool2d(self.kernel_size, self.stride)
        # self.dropout3 = nn.Dropout(0.3)

        # self.conv4 = nn.Conv2d(64, 128, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        # self.dropout4 = nn.Dropout(0.3)

        # self.conv5 = nn.Conv2d(128, 32, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        # self.dropout5 = nn.Dropout(0.3)

        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        # x = self.pooling1(x)
        x = self.dropout1(x)

        x = nn.functional.relu(self.conv2(x))
        # x = self.pooling2(x)
        x = self.dropout2(x)

        x = nn.functional.relu(self.conv3(x))
        # x = self.pooling3(x)
        x = self.dropout3(x)

        # x = nn.functional.relu(self.conv4(x))
        # x = self.dropout4(x)
        # x = nn.functional.relu(self.conv5(x))
        # x = self.dropout5(x)
        x = self.head(x.view(x.size(0), -1))
        return x 


import torch.nn.functional as F

class FullyConnectedQNetwork(nn.Module):
    def __init__(self, input_size, outputs):
        super(FullyConnectedQNetwork, self).__init__()
        
        # Define the size of the hidden layers
        hidden_1 = 60
        output_size = 6
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_1)
        self.fc2 = nn.Linear(hidden_1, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
