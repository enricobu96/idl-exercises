import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=14):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=4)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4)

        self.linear_layer1 = nn.Linear(21632, 128)
        self.linear_layer2 = nn.Linear(128, 96)
        self.linear_layer3 = nn.Linear(96, num_classes)

        # Dropout
        self.dropout = nn.Dropout(p=.5)


    def forward(self, x):

        # Without batch norm
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = torch.flatten(x, 1)

        x = self.linear_layer1(x)
        x = F.sigmoid(x)
        
        x = self.linear_layer2(x)
        x = F.sigmoid(x)

        # Dropout
        x = self.dropout(x)

        x = self.linear_layer3(x)

        return F.sigmoid(x)