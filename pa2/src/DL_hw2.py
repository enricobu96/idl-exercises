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
N_EPOCHS = 40
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 100
LR = 0.05

#--- fixed constants ---
NUM_CLASSES = 24
DATA_DIR = '../data/sign_mnist_%s'

"""
OUR CONSTANTS
- IS_VERBOSE: to avoid too much output
- PATIENCE: the number of previous validation losses smaller than the actual one needed to early stop the training
"""
IS_VERBOSE = True
PATIENCE = 8

# --- Dataset initialization ---
"""
DATA AUGMENTATION. Tried techniques:
- ColorJitter: random brightness and contrast change
- RandomAdjustSharpness: random change sharpness (not too hardly and with low probability)
- RandomInvert: random invert colors
- RandomRotation: just a small rotation

Finally it seems that this configuration works better, but somewhat depends also on how data is (randomly) initialized
"""
train_transform = transforms.Compose([
                                        # transforms.ColorJitter(brightness=.5, contrast=.3),
                                        # transforms.RandomAdjustSharpness(sharpness_factor=1.1, p=.1),
                                        # transforms.RandomInvert(p=.1),
                                        # transforms.RandomRotation(degrees=2),
                                        transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])

train_set = datasets.ImageFolder(DATA_DIR % 'train', transform=train_transform)
dev_set   = datasets.ImageFolder(DATA_DIR % 'dev',   transform=test_transform)
test_set  = datasets.ImageFolder(DATA_DIR % 'test',  transform=test_transform)

# Create Pytorch data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE_TEST, shuffle=True)

# DataLoader for the validation set
valid_loader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=BATCH_SIZE_TEST, shuffle=True)

#--- model ---
class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()
        """
        OUR CNN HERE
        - conv1, conv2: convolutional layers
        - pool, pool2: pooling layers
        - linear_layer1,linear_layer2,linear_layer1: linear layers for the FFNN that learns
        """
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=4)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear_layer1 = nn.Linear(16*4*4, 120)
        self.linear_layer2 = nn.Linear(120, 96)
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

        return self.linear_layer3(x)

#--- set up ---
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = CNN().to(device)

"""
OPTIMIZER(s) AND LOSS FUNCTION
"""
# optimizer = optim.Adam(model.parameters(), lr=LR) #, weight_decay=1e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=LR)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=.2)
# optimizer = torch.optim.Adagrad(model.parameters(), lr=LR)

loss_function = nn.CrossEntropyLoss()

#--- training ---
pre_valid_loss = float('inf')
pre_valid_losses = []
for epoch in range(N_EPOCHS):

    train_loss = 0
    train_correct = 0
    total = 0
    valid_losses = []

    for batch_num, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        """
        TRAINING STEPS
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
    
    """
    EARLY STOPPING
    When validation loss is higher than the previous PATIENCE ones it stops.
    If there have been PATIENCE or more previous losses smaller than the actual, stop
    """
    model.eval() 
    for data, target in valid_loader:
        output = model(data)
        loss = loss_function(output, target)
        valid_losses.append(loss.item())
    
    valid_loss = np.average(valid_losses)
    pre_valid_loss = valid_loss
    pre_valid_losses.append(valid_loss)

    print('Epoch', epoch, 'Validation loss', valid_loss)

    j = 0
    # Now start checking if it has to stop
    if len(pre_valid_losses) >= PATIENCE:
        for l in pre_valid_losses:
            if l < valid_loss:
                j+=1
    if(j>=PATIENCE):
        break

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

"""
REGULARIZATION AND OPTIMIZATION

Before doing any regularization, we started with the following parameters:
- N_EPOCHS = 40
- BATCH_SIZE_TRAIN = 100
- LR = 0.05
- PATIENCE = 40 (to be sure it was deactivated)
- Optimizer: plain SGD without momentum

With these we were able to achieve the following baseline:
- Maximum train accuracy: around 100%
- Test accuracy: between 82% and 86%

Regarding regularization, we tried the following techniques:
- Weight decay: weight_decay=1e-4 parameter in Adam optimizer
- Dropout: dropout in the CNN structure. In this case we used a dropout with probability=15%
- Batch norm: applied in the CNN structure, on the first convolutional layer
- Early stopping: we tried early stopping setting PATIENCE to 6

And we got the following results:

|  Technique | Training accuracy (approx.) | Test accuracy (range) |
| -----------|-----------------------------|-----------------------|
|Weight decay|       98% to 99%            |      82% to 87%       |
|   Dropout  |       97% to 99%            |      79% to 83%       |
| Batch norm |       around 100%           |      80% to 83%       |
| Early stop |       around 90%            |      76% to 82%       |

Regarding optimization, we tried the following techniques (note: we also used early stopping here):
- SGD: stochastic gradient descent without momentum (code commented in optimizer initialization section)
- SGD with momentum: stochastic gradient descent with momentum=.2 (actual code)
- Adam: adam optimizer (code commented in optimizer initialization section)
- AdaGrad: adagrad optimizer (code commented in optimizer initialization section)

And we got the following results:
| Technique | Training accuracy (approx.) | Test accuracy (range) |
|-----------|-----------------------------|-----------------------|
|   SGD     |          around 100%        |       82% to 86%      |
| SGD m=.2  |          around 100%        |       93% to 95%      |
|   Adam    |          around 90%         |       76% to 82%      |
|  AdaGrad  |          99% to 100%        |       85% to 87%      |
"""