import torch
from torch import nn

class QNetwork(nn.Module):
    def __init__(self, h, w, outputs):
        super(QNetwork, self).__init__()
        self.kernel_size = 3
        self.stride = 2
        self.padding = 2

        self.conv1 = nn.Conv2d(5, 16, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.bn6 = nn.BatchNorm2d(64)

        def conv2d_size_out(size, kernel_size=self.kernel_size, stride=self.stride):
            """
            Calculate the output height or width of a convolutional layer
            """
            return (self.padding * 2 + size - kernel_size + stride) // stride

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))))
        linear_input_size = convw * convh * 64
        self.fc1 = nn.Linear(linear_input_size, 128)  # Fully connected layer with 128 units
        self.head = nn.Linear(128, outputs)

    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = nn.functional.relu(self.bn4(self.conv4(x)))
        x = nn.functional.relu(self.bn5(self.conv5(x)))
        x = nn.functional.relu(self.bn6(self.conv6(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = nn.functional.relu(self.fc1(x))  # Apply ReLU activation to the fully connected layer
        return self.head(x)
