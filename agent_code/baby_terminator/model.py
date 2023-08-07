import torch
from torch import nn

class QNetwork(nn.Module):
    def __init__(self, h, w, outputs):
        super(QNetwork, self).__init__()
        self.kernel_size = 3
        self.stride = 1
        self.padding = 2

        self.conv1 = nn.Conv2d(5, 16, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.dropout3 = nn.Dropout2d(p=0.5)
        def conv2d_size_out(size, kernel_size=self.kernel_size, stride=self.stride):
            """
            calculate the Output height or width of a convolutional layer
            """
            return (self.padding * 2 + size - kernel_size + stride) // stride 

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.dropout1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.dropout2(x)
        x = nn.functional.relu(self.conv3(x))
        x = self.dropout3(x)
        return self.head(x.view(x.size(0), -1))
