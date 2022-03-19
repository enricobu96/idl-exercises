"""
   Introduction to Deep Learning
   Assignment 3: Sentiment Classification of Tweets on a Recurrent Neural Network using Pretrained Embeddings

   Hande Celikkanat

   Credit: Data preparation pipeline adopted from https://medium.com/@sonicboom8/sentiment-analysis-torchtext-55fb57b1fab8

   NOTE: we had to change torchtext.data into torchtext.legacy.data due to version compability reasons
"""

import sys
import os
from unicodedata import bidirectional
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
OUR CONSTANTS
- REC_HIDDEN_SIZE: size for the hidden nodes for recurrent layer
- REC_BIDIRECTIONAL: boolean variable, with True the LSTM will be bidirectional
- LR: learning rate for the optimizer
- CL_HIDDEN_SIZE: size for the hidden layer of the FFNN
"""
REC_HIDDEN_SIZE = 20
REC_BIDIRECTIONAL = False
LR = 0.05
CL_HIDDEN_SIZE = 64
BATCH_SIZE = 50

"""
FUNCTIONS
"""
tok = spacy.load('en_core_web_sm',disable=['parser', 'tagger', 'ner'])
# TODO: possibly improve tokenizer
def tokenizer(s): 
    return [w.text.lower() for w in tok(tweet_clean(s))]

def tweet_clean(text):
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    text = re.sub(r'https?:/\/\S+', ' ', text)
    return text.strip()

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

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

"""
Recurrent Network
Params:
    Embedding layer:
    - vocab_size: vocabulary size, i.e. input size for embedding layer, default 10 (random initialize)
    - embedding_dim: embedding dimension for the embedding layer

    Recurrent layer:
    - rec_input_size: input size for recurrent layer, default 200
    - rec_hidden_size: hidden states size for recurrent layer, default REC_HIDDEN_SIZE
    - rec_bidiretional: boolean variable, if True then LSTM is bidirectional, default REC_BIDIRECTIONAL

    Classifier layer:
    - cl_hidden_size: size for the hidden layer of the FFNN, default CL_HIDDEN_SIZE
"""
class RNN(nn.Module):
    def __init__(
        self, vocab_size=10,
        embedding_dim=EMBEDDING_DIM,
        rec_input_size=200,
        rec_hidden_size=REC_HIDDEN_SIZE,
        rec_bidirectional=REC_BIDIRECTIONAL#,
        # cl_hidden_size=CL_HIDDEN_SIZE
        ):
        super().__init__()
        """
        OUR CODE HERE
        """
        self.rec_hidden_size = rec_hidden_size
        self.output_size = 2

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Recurrent layer
        self.lstm = nn.LSTM(rec_input_size, rec_hidden_size, bidirectional=rec_bidirectional)

        # Classifier layer
        self.fc1 = nn.Linear(rec_hidden_size, self.output_size)
 
    def forward(self, x, length):
        out = self.embedding(x) # embedding layer
        out = nn.utils.rnn.pack_padded_sequence(out, length) # packing # TODO possibly remove this
        out, (hidden, cell) = self.lstm(out)
        out, out_length = nn.utils.rnn.pad_packed_sequence(out)
        # NOTE: we can add dropout for normalization
        out = self.fc1(out)
        out = F.log_softmax(out, dim=1)
        return out[0]

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
        vocab_size = len(txt_field.vocab)
        model = RNN(vocab_size=vocab_size)

	    # Copy the pretrained embeddings into the model
        pretrained_embeddings = txt_field.vocab.vectors
        model.embedding.weight.data.copy_(pretrained_embeddings)

	    # Fix the <UNK> and <PAD> tokens in the embedding layer
        model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

        """
        OUR CODE HERE
        """
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
        # criterion = nn.BCELoss()
        criterion = nn.CrossEntropyLoss()

        model = model.to(device)
        criterion = criterion.to(device)

        # --- Train Loop ---
        for epoch in range(N_EPOCHS):
            start_time = time.time()

            epoch_loss = 0
            epoch_acc = 0
            
            model.train()
            
            for batch in train_iter:
                optimizer.zero_grad()
                text, text_lengths = batch.TweetText
                predictions = model(text,text_lengths).squeeze(1)
                loss = criterion(predictions, batch.Label)
                epoch_acc = get_accuracy(predictions, batch.Label)
                loss.backward()
                optimizer.step()
                epoch_loss = loss.item()

            train_loss, train_acc = (epoch_loss / len(train_iter), epoch_acc / len(train_iter)) 
            valid_loss, valid_acc = evaluate(model, dev_iter, criterion)
            
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')