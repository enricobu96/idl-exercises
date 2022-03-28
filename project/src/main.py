from __future__ import annotations
import os
import torch
import warnings
from utils.data_loader import ImageDataset
from model.cnn import CNN
import torch.nn as nn
from torchvision import transforms
import numpy as np
from utils.performance_measure import precision_recall_f1

torch.set_printoptions(threshold=10_000) #TODO: remove
torch.set_num_threads(22)

"""
HYPERPARAMETERS
"""
TRAIN_SIZE = 0.8
BATCH_SIZE_TRAIN = 10
BATCH_SIZE_TEST = 10
LR = .05
N_EPOCHS = 10
PATIENCE = 5
IS_VERBOSE = True
ACTIVATION_TRESHOLD = 0.3
WEIGHT_DECAY = 0.1

"""
SETUP
"""
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

warnings.filterwarnings('ignore')

def collate_fn(batch):
    return tuple(zip(*batch))

# Get all classes
classes = []
for filename in os.listdir('../data/annotations'):
    filename, _ = os.path.splitext(filename)
    classes.append(filename)
classes = list(set(classes))

"""
DATA AUGMENTATION
"""
# train_transform = transforms.Compose([
#                                         transforms.ColorJitter(brightness=.5, contrast=.3),
#                                         transforms.RandomAdjustSharpness(sharpness_factor=1.1, p=.1),
#                                         transforms.RandomInvert(p=.1),
#                                         transforms.RandomRotation(degrees=2)
#                                         ])

"""
DATA LOADING
"""
# Load all data
data = ImageDataset(label_dir='../data/annotations', img_dir='../data/images', classes=classes)#, transform=train_transform)

# Train-test split
train_size = int(TRAIN_SIZE*len(data))
test_size = int(len(data)-train_size)
train_set, test_set = torch.utils.data.random_split(data, [train_size, test_size])

# # If we want also validation set
# test_size = int(test_size/2)
# valid_size = test_size
# train_set, test_set, valid_set = torch.utils.data.random_split(data, [train_size, test_size, valid_size])


# Create loaders
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE_TEST, shuffle=True, collate_fn=collate_fn)

# # If we want also validation set
# valid_loader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=BATCH_SIZE_TEST, shuffle=True, collate_fn=collate_fn)

"""
MODEL INITIALIZATION
"""
model = CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_function = nn.BCELoss()

"""
TRAIN
"""
pre_valid_loss = float('inf')
pre_valid_losses = []
for epoch in range(N_EPOCHS):
    train_loss = 0
    total = 0
    valid_losses = []
    precision = 0
    recall = 0
    f1 = 0

    for batch_num, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data, target = torch.stack(data, dim=0), torch.stack(target, dim=0)
        outputs = model(data.float())
        loss = loss_function(outputs, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predictions = outputs.data
        predictions = torch.argwhere(predictions > ACTIVATION_TRESHOLD)

        target = torch.argwhere(target)
        _precision, _recall, _f1 = precision_recall_f1(predictions, target)
        precision += _precision
        recall += _recall
        f1 += _f1

        if IS_VERBOSE:
            print('Training: Epoch %d - Batch %d/%d: Loss: %.4f' % 
              (epoch, batch_num, len(train_loader), train_loss / (batch_num + 1)))

    print('EPOCH', epoch, 'PRECISION:', (precision / (train_size/BATCH_SIZE_TRAIN)))
    print('EPOCH', epoch, 'RECALL:', (recall / (train_size/BATCH_SIZE_TRAIN)))
    print('EPOCH', epoch, 'F1-SCORE:', (f1 / (train_size/BATCH_SIZE_TRAIN)))

    """
    EARLY STOPPING
    When validation loss is higher than the previous PATIENCE ones it stops.
    If there have been PATIENCE or more previous losses smaller than the actual, stop
    """
    model.eval() 
    for data, target in test_loader:
        data, target = torch.stack(data, dim=0), torch.stack(target, dim=0)
        output = model(data.float())
        loss = loss_function(output, target.float())
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

"""
TEST
"""
test_loss = 0
precision = 0
recall = 0
model.eval()
with torch.no_grad():
    for batch_num, (data, target) in enumerate(test_loader):
        data, target = torch.stack(data, dim=0), torch.stack(target, dim=0)
        outputs = model(data.float())
        loss = loss_function(outputs, target.float())
        test_loss += loss.item()
        predictions = outputs.data
        predictions = torch.argwhere(predictions > ACTIVATION_TRESHOLD)
        target = torch.argwhere(target)

        _precision, _recall, _f1 = precision_recall_f1(predictions, target)
        precision += _precision
        recall += _recall
        f1 += _f1

        if IS_VERBOSE:
            print('Evaluating: Batch %d/%d: Loss: %.4f' % 
              (batch_num, len(test_loader), test_loss / (batch_num + 1)))
    print('TEST PRECISION:', (precision / (test_size/BATCH_SIZE_TEST)))
    print('TEST RECALL:', (recall / (test_size/BATCH_SIZE_TEST)))
    print('TEST F1-SCORE',  (f1 / (train_size/BATCH_SIZE_TEST)))