import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

class CNN4(nn.Module):
    def __init__(self, dropout, num_classes=14):
        super(CNN4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=4)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=4)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=4)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=4)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.linear_layer1 = nn.Linear(37632, 128)
        self.linear_layer2 = nn.Linear(128, 64)
        self.linear_layer3 = nn.Linear(64, num_classes)

        # Dropout
        self.dropout = nn.Dropout(p=.3)
        self.dropout_yes_or_no = dropout

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)

        x = self.pool1(x)
        x = self.pool2(x)

        x = torch.flatten(x, 1)

        x = self.linear_layer1(x)
        x = F.relu(x)
        
        x = self.linear_layer2(x)
        x = F.relu(x)

        # Dropout
        if self.dropout_yes_or_no:
            x = self.dropout(x)

        x = self.linear_layer3(x)

        return F.sigmoid(x)