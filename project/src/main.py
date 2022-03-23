from __future__ import annotations
import os
from torchvision.io import read_image
import torch
import warnings
from utils.data_loader import ImageDataset
from model.cnn import CNN
import torch.nn as nn
import numpy as np

torch.set_printoptions(threshold=10_000) #TODO: remove
torch.set_num_threads(12)


"""
HYPERPARAMETERS
"""
TRAIN_SIZE = 0.6
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 100
LR = .005
N_EPOCHS = 2
PATIENCE = 2
IS_VERBOSE = False
ACTIVATION_TRESHOLD = 0.3

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
DATA LOADING
"""
# Load all data
data = ImageDataset(label_dir='../data/annotations', img_dir='../data/images', classes=classes)

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
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=.1)
loss_function = nn.BCELoss()

"""
TRAIN
"""
pre_valid_loss = float('inf')
pre_valid_losses = []
for epoch in range(N_EPOCHS):
    train_loss = 0
    train_correct = 0
    total = 0
    valid_losses = []

    for batch_num, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data, target = torch.stack(data, dim=0), torch.stack(target, dim=0)
        outputs = model(data.float())
        loss = loss_function(outputs, target.float())
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
        # # For debugging
        # predictions = outputs.data
        # predictions = torch.argwhere(predictions > ACTIVATION_TRESHOLD)
        # target = torch.argwhere(target)
        # print('+++BATCH+++', batch_num, '\n PREDICITONS', outputs.data, '\n TARGET', target)

"""
TEST
"""
model.eval()
with torch.no_grad():
    for batch_num, (data, target) in enumerate(test_loader):
        data, target = torch.stack(data, dim=0), torch.stack(target, dim=0)
        outputs = model(data.float())
        predictions = outputs.data
        predictions = torch.argwhere(predictions > ACTIVATION_TRESHOLD)
        target = torch.argwhere(target)
        print('PREDICTIONS', predictions, 'TARGET', target)
        # print('PREDICTIONS', outputs.data, 'TARGET', target)





















# TODO: to add to training part
    #     train_correct += int(sum(predictions == target))
    #     total += target.size(0)

    #     if IS_VERBOSE:
    #         print('Training: Epoch %d - Batch %d/%d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' % 
    #           (epoch, batch_num, len(train_loader), train_loss / (batch_num + 1), 
    #            100. * train_correct / total, train_correct, total))
    
    # """
    # EARLY STOPPING
    # When validation loss is higher than the previous PATIENCE ones it stops.
    # If there have been PATIENCE or more previous losses smaller than the actual, stop
    # """
    # model.eval() 
    # for data, target in valid_loader:
    #     output = model(data)
    #     loss = loss_function(output, target)
    #     valid_losses.append(loss.item())
    
    # valid_loss = np.average(valid_losses)
    # pre_valid_loss = valid_loss
    # pre_valid_losses.append(valid_loss)

    # print('Epoch', epoch, 'Validation loss', valid_loss)

    # j = 0
    # # Now start checking if it has to stop
    # if len(pre_valid_losses) >= PATIENCE:
    #     for l in pre_valid_losses:
    #         if l < valid_loss:
    #             j+=1
    # if(j>=PATIENCE):
    #     break

