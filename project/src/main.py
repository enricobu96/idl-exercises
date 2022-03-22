from __future__ import annotations
import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
import warnings
import re
from utils.data_loader import ImageDataset

"""
SETUP
"""
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

warnings.filterwarnings('ignore')

"""
HYPERPARAMETERS
"""
TRAIN_SIZE = 0.8
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 100

"""
DATA LOADING
"""
# Load all data
data = ImageDataset(label_dir='../data/annotations', img_dir='../data/images')

# Train-test-dev split
train_size = int(TRAIN_SIZE*len(data))
test_size = int((len(data)-train_size)/2)
valid_size = test_size
train_set, test_set, dev_set = torch.utils.data.random_split(data, [train_size, test_size, valid_size])

# Create loaders
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE_TEST, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=BATCH_SIZE_TEST, shuffle=True)

for batch_num, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        print(data, target)
        break