import csv
import sys

import nltk
# Make sure that we've downloaded the tokenizer.
# nltk.download("punkt")
from nltk.tokenize import TweetTokenizer

SEMEVALHEADER = ['ID','SENTIMENT','BODY']
TOKENIZER = TweetTokenizer()
BIAS = 'BIAS'
LABEL_INDICES = {'negative':0, 'neutral':1, 'positive':2}

def read_semeval(filename):
    """
    Read a list of tweets with sentiment labels from @sefilename. Each
    tweet is a dictionary with keys:
 
    ID        - ID number of tweet.
    SENTIMENT - Sentiment label for this tweet.
    BODY      - List of tokens of this tweet.

    """
    data = []

    with open(filename, encoding='utf-8') as sefile:
        csvreader = csv.reader(sefile, delimiter='\t', quoting=csv.QUOTE_NONE)
        for i, fields in enumerate(csvreader):
            if fields and len(fields) != len(SEMEVALHEADER):
                raise SyntaxError('Incorrect field count', 
                                  (filename, i, None, None)) 
            tweet = dict(zip(SEMEVALHEADER,fields))
            tweet["BODY"] = TOKENIZER.tokenize(tweet["BODY"].lower()) + [BIAS]
            data.append(tweet)
    return data

def read_semeval_datasets(data_dir):
    data = {}
    for data_set in ["training","development.input","development.gold",
                     "test.input","test.gold"]:
        data[data_set] = read_semeval("%s/%s.txt" % (data_dir,data_set)) 
    return data

def write_semeval(data,output,output_file):
    for ex, klass in zip(data,output):
        print("%s\t%s\t%s" % (ex["ID"], klass, ex["ORIG_BODY"]),
              file=output_file)

if __name__=="__main__":
    # Check that we don't crash on reading.
    read_semeval('%s/training.txt' % sys.argv[1])
    read_semeval('%s/development.input.txt' % sys.argv[1])
    read_semeval('%s/development.gold.txt' % sys.argv[1])
    read_semeval('%s/test.input.txt' % sys.argv[1])
    read_semeval('%s/test.gold.txt' % sys.argv[1])

