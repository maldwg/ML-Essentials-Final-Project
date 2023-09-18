import torch
from torch import nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, h, w, outputs):
        super(QNetwork, self).__init__()
        self.kernel_size = 3
        self.stride = 1
        self.padding = 2

        self.conv1 = nn.Conv2d(3, 16, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        # self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        #self.pooling2 = nn.MaxPool2d(self.kernel_size, self.stride)
        # self.dropout2 = nn.Dropout(0.3)

        # self.conv3 = nn.Conv2d(32, 64, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        # self.pooling3 = nn.MaxPool2d(self.kernel_size, self.stride)
        # self.dropout3 = nn.Dropout(0.3)

        # self.conv4 = nn.Conv2d(64, 128, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        # self.dropout4 = nn.Dropout(0.3)

        # self.conv5 = nn.Conv2d(128, 32, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        # self.dropout5 = nn.Dropout(0.3)

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
        
        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))
        linear_input_size = convw * convh * 32

        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        # x = self.pooling1(x)
        # x = self.dropout1(x)

        x = nn.functional.relu(self.conv2(x))
        # x = self.pooling2(x)
        # x = self.dropout2(x)

        # x = nn.functional.relu(self.conv3(x))
        # x = self.pooling3(x)
        # x = self.dropout3(x)

        # x = nn.functional.relu(self.conv4(x))
        # x = self.dropout4(x)
        # x = nn.functional.relu(self.conv5(x))
        # x = self.dropout5(x)

        x = self.head(x.view(x.size(0), -1))
        x = F.softmax(x)
        return x 


