from torch import nn
import torch.nn.functional as F


class PolicyGradientNetwork(nn.Module):
    """
    Neural network architecture for the Policy Gradient model.

    :param h: Height of the input image.
    :param w: Width of the input image.
    :param outputs: Number of output classes (actions).
    """

    def __init__(self, h, w, outputs):
        super(PolicyGradientNetwork, self).__init__()

        # Define convolutional layer parameters
        self.kernel_size = 3
        self.stride = 1
        self.padding = 2

        # First convolutional layer with 3 input channels and 16 output channels
        self.conv1 = nn.Conv2d(
            3,
            16,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

        # Second convolutional layer with 16 input channels and 32 output channels
        self.conv2 = nn.Conv2d(
            16,
            32,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

        def conv2d_size_out(
            size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
        ):
            """
            Calculate the output height or width of a convolutional layer.

            :param size: Input height or width.
            :param kernel_size: Size of the convolutional kernel.
            :param stride: Stride of the convolutional layer.
            :param padding: Padding of the convolutional layer.

            :return: int: Output height or width.
            """
            return (padding * 2 + size - kernel_size + stride) // stride

        # Calculate the size of the output from the final convolutional layer
        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))
        linear_input_size = convw * convh * 32

        # Fully connected layer
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        """
        Forward pass of the network.

        :param x: Input tensor.
        :return: Softmax output tensor.
        """
        # Apply first convolutional layer with ReLU activation
        x = nn.functional.relu(self.conv1(x))
        # Apply second convolutional layer with ReLU activation
        x = nn.functional.relu(self.conv2(x))
        # Flatten and apply fully connected layer
        x = self.head(x.view(x.size(0), -1))
        # Apply softmax to get probabilities
        return F.softmax(x, dim=-1)
