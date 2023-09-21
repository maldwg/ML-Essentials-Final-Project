from torch import nn
import torch.nn.functional as F

class PolicyGradientNetwork(nn.Module):
    def __init__(self, h, w, outputs):
        super(PolicyGradientNetwork, self).__init__()
        self.kernel_size = 3
        self.stride = 1
        self.padding = 2

        self.conv1 = nn.Conv2d(
            3,
            16,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

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
            calculate the Output height or width of a convolutional layer
            """
            return (padding * 2 + size - kernel_size + stride) // stride


        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))
        linear_input_size = convw * convh * 32

        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.head(x.view(x.size(0), -1))
        return F.softmax(x, dim=-1)
        