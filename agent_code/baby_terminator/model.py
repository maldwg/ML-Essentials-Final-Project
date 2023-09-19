from torch import nn

class Convolution(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dropout=0):
        super(Convolution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout = dropout
        self.output_channels = output_channels
        
        self.conv = nn.Conv2d(input_channels, self.output_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        x = self.dropout(x)
        return nn.functional.leaky_relu(self.conv(x))


class LinearWithDropout(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super(LinearWithDropout, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class QNetwork(nn.Module):
    def __init__(self, h, w, outputs, convolutions, head_dropout):
        super(QNetwork, self).__init__()
        self.convolutions = nn.ModuleList(convolutions)
        self.kernel_size = self.convolutions[-1].kernel_size
        self.stride = self.convolutions[-1].stride
        self.padding = self.convolutions[-1].padding

        def conv2d_size_out(size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding):
            """
            calculate the Output height or width of a convolutional layer
            """
            return (padding * 2 + size - kernel_size + stride) // stride 
        
        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))
        linear_input_size = convw * convh * self.convolutions[-1].output_channels

        self.head = LinearWithDropout(linear_input_size, outputs, head_dropout)

    def forward(self, x):
        for conv in self.convolutions:
            x = conv(x)

        x = self.head(x.view(x.size(0), -1))
        return x 

    class Builder:
        def __init__(self):
            self.convolutions = []
            self.h = None
            self.w = None
            self.channels = None
            self.out_dim = None
            self.head_dropout = None

        def input_output_dimensions(self, h, w, channels, out_dim):
            self.h = h
            self.w = w
            self.channels = channels
            self.out_dim = out_dim
            return self

        def set_head_dropout(self, head_dropout):
            self.head_dropout = head_dropout
            return self
        
        def add_convolution(self, output_channels, kernel_size, stride, padding, dropout=0):
            if len(self.convolutions) == 0:
                self.convolutions.append(Convolution(self.channels, output_channels, kernel_size, stride, padding, dropout))
            else:
                self.convolutions.append(Convolution(self.convolutions[-1].output_channels, output_channels, kernel_size, stride, padding, dropout))
            return self

        def is_valid_network(self):
            valid_convolutions = len(self.convolutions) > 0
            valid_dimensions = self.h is not None and self.w is not None and self.channels is not None and self.out_dim is not None
            valid_head = self.head_dropout is not None
            return valid_convolutions and valid_dimensions and valid_head

        def build(self):
            if self.is_valid_network():
                return QNetwork(self.h, self.w, self.out_dim, self.convolutions, self.head_dropout)
            else:
                raise ValueError("One or more values for creating a QNetwork are missing!")

        
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
