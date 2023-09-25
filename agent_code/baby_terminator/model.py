from torch import nn


class Convolution(nn.Module):
    """
    A convolutional layer with dropout and leaky ReLU activation.

    :param input_channels : int, Number of input channels.
    :param output_channels : int, Number of output channels.
    :param kernel_size : int, Size of the convolutional kernel.
    :param stride : int, Stride of the convolution.
    :param padding : int,  Padding for the convolution.
    :param dropout : float, optional Dropout rate, default is 0.

    :attribute conv : nn.Conv2d, The convolutional layer.
    :attribute dropout : nn.Dropout, The dropout layer.

    """

    def __init__(
        self, input_channels, output_channels, kernel_size, stride, padding, dropout=0
    ):
        super(Convolution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout = dropout
        self.output_channels = output_channels

        self.conv = nn.Conv2d(
            input_channels,
            self.output_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        """
        Forward pass for the Convolution layer.

        :param x : torch.Tensor, Input tensor.

        :return: torch.Tensor, Output tensor after applying convolution, dropout, and activation.
        """
        x = self.dropout(x)
        return nn.functional.leaky_relu(self.conv(x))


class LinearWithDropout(nn.Module):
    """
    A linear layer with dropout.

    :param input_size : int, Number of input features.
    :param output_size : int, Number of output features.
    :param dropout : float, Dropout rate.

    :attribute dropout : nn.Dropout, The dropout layer.
    :attribute linear : nn.Linear, The linear layer.
    """

    def __init__(self, input_size, output_size, dropout):
        super(LinearWithDropout, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        """
        :param x: Input tensor.

        :return: Output tensor after applying linear transformation and dropout.
        """
        x = self.dropout(x)
        return self.linear(x)


class QNetwork(nn.Module):
    """
    :param h: Height of the input.
    :param w: Width of the input.
    :param outputs: Number of output units.
    :param convolutions: List of Convolution modules.
    :param head_dropout: Dropout rate for the head (final) layer.

    :return None: Initializes the QNetwork's attributes.
    """

    def __init__(self, h, w, outputs, convolutions, head_dropout):
        super(QNetwork, self).__init__()
        self.convolutions = nn.ModuleList(convolutions)
        self.kernel_size = self.convolutions[-1].kernel_size
        self.stride = self.convolutions[-1].stride
        self.padding = self.convolutions[-1].padding

        def conv2d_size_out(
            size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
        ):
            """
            Calculates the output height or width of a convolutional layer.

            :param size: Size of the input (height or width).
            :param kernel_size: Size of the convolutional kernel.
            :param stride: Stride of the convolution.
            :param padding: Padding for the convolution.

            :return: Output size (height or width) after the convolution.
            """
            return (padding * 2 + size - kernel_size + stride) // stride

        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))
        linear_input_size = convw * convh * self.convolutions[-1].output_channels

        self.head = LinearWithDropout(linear_input_size, outputs, head_dropout)

    def forward(self, x):
        """
        :param x: Input tensor.

        :return: Output tensor after passing through the network.
        """
        for conv in self.convolutions:
            x = conv(x)

        x = self.head(x.view(x.size(0), -1))
        return x

    class Builder:
        def __init__(self):
            """
            :return None: Initializes the Builder class's attributes.
            """
            self.convolutions = []
            self.h = None
            self.w = None
            self.channels = None
            self.out_dim = None
            self.head_dropout = None

        def input_output_dimensions(self, h, w, channels, out_dim):
            """
            Sets the dimensions for the input and output of the network.

            :param h: Height of the input.
            :param w: Width of the input.
            :param channels: Number of input channels.
            :param out_dim: Number of output dimensions.

            :return: Returns the Builder object for method chaining.
            """
            self.h = h
            self.w = w
            self.channels = channels
            self.out_dim = out_dim
            return self

        def set_head_dropout(self, head_dropout):
            """
            Sets the dropout rate for the head (final) layer.

            :param head_dropout: Dropout rate for the head layer.

            :return: Returns the Builder object for method chaining.
            """
            self.head_dropout = head_dropout
            return self

        def add_convolution(
            self, output_channels, kernel_size, stride, padding, dropout=0
        ):
            """
            Adds a convolution layer to the network.

            :param output_channels: Number of output channels.
            :param kernel_size: Size of the convolutional kernel.
            :param stride: Stride of the convolution.
            :param padding: Padding for the convolution.
            :param dropout: Dropout rate, optional (default is 0).

            :return: Returns the Builder object for method chaining.
            """
            if len(self.convolutions) == 0:
                self.convolutions.append(
                    Convolution(
                        self.channels,
                        output_channels,
                        kernel_size,
                        stride,
                        padding,
                        dropout,
                    )
                )
            else:
                self.convolutions.append(
                    Convolution(
                        self.convolutions[-1].output_channels,
                        output_channels,
                        kernel_size,
                        stride,
                        padding,
                        dropout,
                    )
                )
            return self

        def is_valid_network(self):
            """
            :return: True if the network can be built, False otherwise.
            """
            valid_convolutions = len(self.convolutions) > 0
            valid_dimensions = (
                self.h is not None
                and self.w is not None
                and self.channels is not None
                and self.out_dim is not None
            )
            valid_head = self.head_dropout is not None
            return valid_convolutions and valid_dimensions and valid_head

        def build(self):
            """
            :return: The built QNetwork if the network can be built.

            :raises ValueError: If the network cannot be built with the current parameters.
            """
            if self.is_valid_network():
                return QNetwork(
                    self.h, self.w, self.out_dim, self.convolutions, self.head_dropout
                )
            else:
                raise ValueError(
                    "One or more values for creating a QNetwork are missing!"
                )
