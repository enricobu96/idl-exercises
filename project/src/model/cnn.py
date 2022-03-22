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
        self.linear_layer1 = nn.Linear(13456, 128)
        self.linear_layer2 = nn.Linear(128, 96)
        self.linear_layer3 = nn.Linear(96, num_classes)

        # # Dropout
        # self.dropout = nn.Dropout(p=.15)

        # # Batch norm
        # self.batch_norm = nn.BatchNorm2d(6)

    def forward(self, x):

        # Without batch norm
        x = self.pool(F.relu(self.conv1(x)))

        # # With batch norm
        # x = self.pool(F.relu(self.batch_norm(self.conv1(x))))

        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.linear_layer1(x))
        x = F.relu(self.linear_layer2(x))

        # # Dropout
        # x = self.dropout(x)

        return F.softmax(self.linear_layer3(x))