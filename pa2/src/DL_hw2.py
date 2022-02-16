from importlib.machinery import DEBUG_BYTECODE_SUFFIXES
import torch
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


#--- hyperparameters ---
N_EPOCHS = 10
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 100
LR = 0.01

#--- fixed constants ---
NUM_CLASSES = 24
DATA_DIR = '../data/sign_mnist_%s'

# OUR CONSTANTS
IS_VERBOSE = True

# --- Dataset initialization ---

# We transform image files' contents to tensors
# Plus, we can add random transformations to the training data if we like
# Think on what kind of transformations may be meaningful for this data.
# Eg., horizontal-flip is definitely a bad idea for sign language data.
# You can use another transformation here if you find a better one.
train_transform = transforms.Compose([
                                        #transforms.RandomHorizontalFlip(), # TODO: add data augmentation
                                        transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])

train_set = datasets.ImageFolder(DATA_DIR % 'train', transform=train_transform)
dev_set   = datasets.ImageFolder(DATA_DIR % 'dev',   transform=test_transform)
test_set  = datasets.ImageFolder(DATA_DIR % 'test',  transform=test_transform)

# Create Pytorch data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE_TEST, shuffle=False)

"""
OUR CODE HERE
"""
valid_loader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=BATCH_SIZE_TEST, shuffle=False)


#--- model ---
class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()
        """
        OUR CODE HERE
        """
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=4)
        self.linear_layer1 = nn.Linear(16*4*4, 120)
        self.linear_layer2 = nn.Linear(120, 84)
        self.linear_layer3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.linear_layer1(x))
        x = F.relu(self.linear_layer2(x))
        return self.linear_layer3(x)

#--- set up ---
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = CNN().to(device)

"""
OUR CODE HERE
"""
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_function = nn.CrossEntropyLoss()

pre_valid_loss = float('inf')

#--- training ---
for epoch in range(N_EPOCHS):

    train_loss = 0
    train_correct = 0
    total = 0
    valid_losses = []

    for batch_num, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        """
        OUR CODE HERE
        """
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_function(outputs, target)
        loss.backward()
        optimizer.step()

        train_loss+=loss.item()
        scores, predictions = torch.max(outputs.data, 1)
        train_correct += int(sum(predictions == target))
        total += target.size(0)

        if IS_VERBOSE:
            print('Training: Epoch %d - Batch %d/%d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' % 
              (epoch, batch_num, len(train_loader), train_loss / (batch_num + 1), 
               100. * train_correct / total, train_correct, total))
    
    # Really simple early stopping on validation loss
    model.eval() 
    for data, target in valid_loader:
        output = model(data)
        loss = loss_function(output, target)
        valid_losses.append(loss.item())
    
    valid_loss = np.average(valid_losses)

    print('Epoch', epoch, 'Validation loss', valid_loss)

    # if pre_valid_loss < valid_loss:  # TODO: implement proper early stopping
    #     break

    # pre_valid_loss = valid_loss


#--- test ---
test_loss = 0
test_correct = 0
total = 0

model.eval()

with torch.no_grad():
    for batch_num, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        test_correct += (predicted == target).sum().item()

        loss = loss_function(outputs, target)
        
        test_loss += loss.item()


        print('Evaluating: Batch %d/%d: Loss: %.4f | Test Acc: %.3f%% (%d/%d)' % 
              (batch_num, len(test_loader), test_loss / (batch_num + 1), 
               100. * test_correct / total, test_correct, total))

