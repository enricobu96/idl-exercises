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
from contextlib import redirect_stdout

torch.set_printoptions(threshold=10_000) #TODO: remove
torch.set_num_threads(22)

def execute(batch_size_train=10, batch_size_test=10, lr=.05, epochs=10, patience=5, activation=0.3, weight_decay=0.1, transform=True, dropout=False):
    """
    HYPERPARAMETERS
    """
    TRAIN_SIZE = 0.8
    BATCH_SIZE_TRAIN = batch_size_train
    BATCH_SIZE_TEST = batch_size_test
    LR = lr
    N_EPOCHS = epochs
    PATIENCE = patience
    IS_VERBOSE = False
    ACTIVATION_TRESHOLD = activation
    WEIGHT_DECAY = weight_decay

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

    # Get all the classes for one-hot encoding
    classes = []
    for filename in os.listdir('../data/annotations'):
        filename, _ = os.path.splitext(filename)
        classes.append(filename)
    classes = list(set(classes))

    """
    DATA AUGMENTATION
    """
    train_transform = transforms.Compose([
                                            transforms.ColorJitter(brightness=.5, contrast=.3),
                                            transforms.RandomAdjustSharpness(sharpness_factor=1.1, p=.1),
                                            transforms.RandomInvert(p=.1),
                                            transforms.RandomRotation(degrees=2)
                                            ]) if transform else None

    """
    DATA LOADING
    """
    # Load all data
    data = ImageDataset(label_dir='../data/annotations', img_dir='../data/images', classes=classes, transform=train_transform)

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
    model = CNN(dropout=dropout).to(device)
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

        # print('EPOCH', epoch, 'PRECISION:', (precision / (train_size/BATCH_SIZE_TRAIN)))
        # print('EPOCH', epoch, 'RECALL:', (recall / (train_size/BATCH_SIZE_TRAIN)))
        # print('EPOCH', epoch, 'F1-SCORE:', (f1 / (train_size/BATCH_SIZE_TRAIN)))

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

        # print('Epoch', epoch, 'Validation loss', valid_loss)

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
        # print('TEST PRECISION:', (precision / (test_size/BATCH_SIZE_TEST)))
        # print('TEST RECALL:', (recall / (test_size/BATCH_SIZE_TEST)))
        # print('TEST F1-SCORE',  (f1 / (train_size/BATCH_SIZE_TEST)))
        print((precision / (test_size/BATCH_SIZE_TEST)), '\t', (recall / (test_size/BATCH_SIZE_TEST)), '\t', (f1 / (test_size/BATCH_SIZE_TEST)))

# Execution with different parameters
batch_sizes = [10, 100, 1000]
lrs = [0.05, 0.1]
epochss = [8]
patiences = [5]
activations = [0.2, 0.3, 0.4]
weight_decays = [0.1, 0.5]
transformms = [True, False]
dropouts = [False, True]

with open('results.txt', 'a') as f:
    with redirect_stdout(f):
        for batch_size in batch_sizes:
            for lr in lrs:
                for epochs in epochss:
                    for patience in patiences:
                        for activation in activations:
                            for weight_decay in weight_decays:
                                for transformm in transformms:
                                    for dropout in dropouts:
                                        print('BATCH SIZE:', batch_size)
                                        print('LR:', lr)
                                        print('EPOCHS:', epochs)
                                        print('PATIENCE:', patience)
                                        print('ACTIVATION:', activation)
                                        print('WEIGHT DECAY:', weight_decay)
                                        print('TRANSFORM:', transformm)
                                        print('DROPOUT:', dropout)
                                        print('Test precision\tTest recall\tTest f1')
                                        execute(batch_size, batch_size, lr, epochs, patience, activation, weight_decay, transformm, dropout)
                                        print('\n')
                                        f.flush()
    f.close()




# # Changing batch sizes
# with open('out.txt', 'a') as f:
#     with redirect_stdout(f):
#         print('CHANGING ONLY BATCH SIZES')
#         for b in batch_sizes:
#             print('\n Batch_size:', b)
#             execute(batch_size_train=b, batch_size_test=b, lr=0.1, epochs=1, patience=1, activation=0.5, weight_decay=0.1, transform=True)
#             f.flush()
#     f.close()

# # Changing LR
# with open('out.txt', 'a') as f:
#     with redirect_stdout(f):
#         print('CHANGING ONLY LR')
#         for b in lrs:
#             print('\n LR:', b)
#             execute(batch_size_train=100, batch_size_test=100, lr=b, epochs=1, patience=1, activation=0.5, weight_decay=0.01, transform=True)
#             f.flush()
#     f.close()
