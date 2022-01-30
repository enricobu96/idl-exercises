"""
   Introduction to Deep Learning (LDA-T3114)
   Skeleton Code for Assignment 1: Sentiment Classification on a Feed-Forward Neural Network

   Hande Celikkanat & Miikka Silfverberg
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

#ATTENTION: If necessary, add the paths to your data_semeval.py and paths.py here:
#import sys
#sys.path.append('</path/to/below/modules>')
from data_semeval import *
from paths import data_dir


#--- hyperparameters ---

N_CLASSES = len(LABEL_INDICES)
N_EPOCHS = 10
LEARNING_RATE = 0.05
BATCH_SIZE = 1
REPORT_EVERY = 1
IS_VERBOSE = True

# Hyperparameters we added
HIDDEN_SIZE_1 = 8
HIDDEN_SIZE_2 = None
SGD_MOMENTUM = .9


def make_bow(tweet, indices):
    feature_ids = list(indices[tok] for tok in tweet['BODY'] if tok in indices)
    bow_vec = torch.zeros(len(indices))
    bow_vec[feature_ids] = 1
    return bow_vec.view(1, -1)

def generate_bow_representations(data):
    vocab = set(token for tweet in data['training'] for token in tweet['BODY'])
    vocab_size = len(vocab) 
    indices = {w:i for i, w in enumerate(vocab)}
  
    for split in ["training","development.input","development.gold",
                  "test.input","test.gold"]:
        for tweet in data[split]:
            tweet['BOW'] = make_bow(tweet,indices)

    return indices, vocab_size

# Convert string label to pytorch format.
def label_to_idx(label):
    return torch.LongTensor([LABEL_INDICES[label]])



#--- model ---

class FFNN(nn.Module):
    # Feel free to add whichever arguments you like here.
    def __init__(self, vocab_size, n_classes, extra_arg_1=8, extra_arg_2=None): #TODO: change default arguments for hidden layers
        super(FFNN, self).__init__()

        self.second_layer = False

        # Arguments initialization
        self.input_size = vocab_size
        self.hidden_size_1 = extra_arg_1
        self.hidden_size_2 = extra_arg_2
        self.output_size = n_classes

        """
        Layers initialization:
            - fc1 = linear function 1: input_size -> hidden_size_1
            - relu1 = first non-linearity layer
        """
        # Input layer and first hidden layer
        self.fc1 = nn.Linear(self.input_size, self.hidden_size_1
        )
        self.relu1 = nn.ReLU()

        # If extra_arg_2: second hidden layer
        if self.hidden_size_2:
            # Second hidden layer
            self.second_layer = True
            self.fc2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
            self.relu2 = nn.reLU()

            # Output layer
            self.out = nn.Linear(self.hidden_size_2, self.output_size)
        # If not extra_arg_2: only one hidden layer
        else:
            self.out = nn.Linear(self.hidden_size_1, self.output_size)


    def forward(self, x):
        # Input layer and first hidden layer
        output = self.fc1(x)
        output = self.relu1(output)

        # If second hidden layer
        if self.second_layer:
            output = self.fc2(output)
            output = self.relu2(output)

        # If not second hidden layer
        output = self.out(output)
        return F.log_softmax(output, dim=1) # Not sure dim should be 1

#--- data loading ---
data = read_semeval_datasets(data_dir)
indices, vocab_size = generate_bow_representations(data)

#--- set up ---

# model extra parameters: up to two hidden layers sizes
model = FFNN(vocab_size, N_CLASSES, HIDDEN_SIZE_1, HIDDEN_SIZE_2)
loss_function = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=SGD_MOMENTUM)


#--- training ---
for epoch in range(N_EPOCHS):
    total_loss = 0
    # Generally speaking, it's a good idea to shuffle your
    # datasets once every epoch.
    random.shuffle(data['training'])    

    for i in range(int(len(data['training'])/BATCH_SIZE)):
        minibatch = data['training'][i*BATCH_SIZE:(i+1)*BATCH_SIZE]

        optimizer.zero_grad()
        probs = model(minibatch[0]['BOW'])
        target = label_to_idx(minibatch[0]['SENTIMENT'])
        loss = loss_function(probs, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
                              
    if ((epoch+1) % REPORT_EVERY) == 0:
        print('epoch: %d, loss: %.4f' % (epoch+1, total_loss*BATCH_SIZE/len(data['training'])))



# #--- test ---
# correct = 0
# with torch.no_grad():
#     for tweet in data['test.gold']:
#         gold_class = label_to_idx(tweet['SENTIMENT'])

#         # WRITE CODE HERE
#         # You can, but for the sake of this homework do not have to,
#         # use batching for the test data.
#         predicted = -1

#         if IS_VERBOSE:
#             print('TEST DATA: %s, GOLD LABEL: %s, GOLD CLASS %d, OUTPUT: %d' % 
#                  (' '.join(tweet['BODY'][:-1]), tweet['SENTIMENT'], gold_class, predicted))

#     print('test accuracy: %.2f' % (100.0 * correct / len(data['test.gold'])))

