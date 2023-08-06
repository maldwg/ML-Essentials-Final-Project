import torch
from torch import nn

class QNetwork(nn.Module):
    def __init__(self, h, w, outputs):
        super(QNetwork, self).__init__()
        self.kernel_size = 3
        self.stride = 1

        self.conv1 = nn.Conv2d(5, 16, kernel_size=self.kernel_size, stride=self.stride)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=self.kernel_size, stride=self.stride)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=self.kernel_size, stride=self.stride)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size = self.kernel_size, stride = self.stride):
            """
            calculate the Output height or width of a convolutional layer
            """
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
