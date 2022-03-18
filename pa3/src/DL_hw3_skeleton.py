"""
   Introduction to Deep Learning
   Assignment 3: Sentiment Classification of Tweets on a Recurrent Neural Network using Pretrained Embeddings

   Hande Celikkanat

   Credit: Data preparation pipeline adopted from https://medium.com/@sonicboom8/sentiment-analysis-torchtext-55fb57b1fab8

   NOTE: we had to change torchtext.data into torchtext.legacy.data due to version compability reasons
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
import spacy
import regex as re
from torchtext import vocab
import time

# ---constants & hyperparameters---
N_EPOCHS = 5
EMBEDDING_DIM = 200
OUTPUT_DIM = 2

"""
FUNCTIONS
"""
# Auxilary functions for data preparation
tok = spacy.load('en_core_web_sm',disable=['parser', 'tagger', 'ner'])
def tokenizer(s): 
    return [w.text.lower() for w in tok(tweet_clean(s))]

def tweet_clean(text):
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    text = re.sub(r'https?:/\/\S+', ' ', text)
    return text.strip()

# Evaluation functions
def get_accuracy(output, gold):
    _, predicted = torch.max(output, dim=1)
    correct = torch.sum(torch.eq(predicted,gold)).item()
    acc = correct / gold.shape[0]
    return acc

def evaluate(model, iterator, criterion):    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.TweetText
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.Label)
            acc = get_accuracy(predictions, batch.Label)
            epoch_loss += loss.item()
            epoch_acc += acc
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# Utility
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

"""
Recurrent Network
"""
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM, start_embeddings=nn.Embedding(10,3)):
        super().__init__()
        """
        OUR CODE HERE
        """
        self.embedding = start_embeddings
        # Embedding layer
            # No idea, but
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        input_size = 0 # TODO: something something

        # Recurrent layer
        # self.rnn = nn.RNN(input_size, rnn_hidden_dim, n_layers)
        
 
    def forward(self):
        """
        OUR CODE HERE
        """
        return 0

if __name__ == '__main__':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # --- Data Preparation ---

        # define the columns that we want to process and how to process
        txt_field = torchtext.legacy.data.Field(sequential=True, 
                                         tokenize=tokenizer, 
                                         include_lengths=True, 
                                         use_vocab=True)
        label_field = torchtext.legacy.data.Field(sequential=False, 
                                           use_vocab=False) 

        csv_fields = [
            ('Label', label_field), # process this field as the class label
            ('TweetID', None), # we dont need this field
            ('Timestamp', None), # we dont need this field
            ('Flag', None), # we dont need this field
            ('UseerID', None), # we dont need this field
            ('TweetText', txt_field) # process it as text field
        ]

        train_data, dev_data, test_data = torchtext.\
            legacy.data.TabularDataset.splits(path='../data',
                                            format='csv', 
                                            train='sent140.train.mini.csv', # TODO: change to only train (big file)
                                            validation='sent140.dev.csv', 
                                            test='sent140.test.csv', 
                                            fields=csv_fields, 
                                            skip_header=False)


        txt_field.build_vocab(train_data, dev_data, max_size=100000, 
                              vectors='glove.twitter.27B.200d', unk_init = torch.Tensor.normal_)
        label_field.build_vocab(train_data)

        train_iter, dev_iter, test_iter = torchtext.legacy.data.BucketIterator.splits(datasets=(train_data, dev_data, test_data), 
                                                    batch_sizes=(50,50,50),  # batch sizes of train, dev, test
                                                    sort_key=lambda x: len(x.TweetText), # how to sort text
                                                    device=device,
                                                    sort_within_batch=True, 
                                                    repeat=False)

        # --- Model, Loss, Optimizer Initialization ---        

        PAD_IDX = txt_field.vocab.stoi[txt_field.pad_token]
        UNK_IDX = txt_field.vocab.stoi[txt_field.unk_token]

        """
        OUR CODE HERE
        """
        model = RNN.to(device)

	    # Copy the pretrained embeddings into the model
        pretrained_embeddings = txt_field.vocab.vectors
        model.embedding.weight.data.copy_(pretrained_embeddings)

	    # Fix the <UNK> and <PAD> tokens in the embedding layer
        model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)


        # WRITE CODE HERE
        optimizer = None
        criterion = None

        model = model.to(device)
        criterion = criterion.to(device)


        # --- Train Loop ---
        for epoch in range(N_EPOCHS):
            start_time = time.time()
            epoch_loss = 0
            epoch_acc = 0
            
            model.train()
            
            for batch in train_iter:
                # WRITE CODE HERE
                pass

            train_loss, train_acc = (epoch_loss / len(train_iter), epoch_acc / len(train_iter)) 
            valid_loss, valid_acc = evaluate(model, dev_iter, criterion)
            
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')