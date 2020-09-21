import pandas as pd
import numpy as np
import re
import argparse
from tqdm import tqdm

from nltk.corpus import stopwords


from gensim.models import Phrases
from gensim.models.phrases import Phraser

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

def loadData(filepath):
    # Retrieving paragraphs : Assumption is that each paragraph in dataset is
    # separated by new line character
    paragraphs = []
    with open(filepath, encoding="utf-8") as f:
        for para in f.readlines():
            if(len(para.strip()) > 0):
                paragraphs.append(para.strip())

    return paragraphs

# pre processing data
def cleanParagraph(paragraph, stopWords):
    processedList = ""
    
    # convert to lowercase, ignore all special characters - keep only alpha-numericals and spaces (not removing full-stop here)
    paragraph = re.sub(r'[^A-Za-z0-9\s.]',r'',str(paragraph).lower())
    paragraph = re.sub(r'\n',r' ',paragraph)
    
    # remove stop words
    paragraph = " ".join([word for word in paragraph.split() if word not in stopWords])
    
    return paragraph

def train(filepath):
    paragraphs = loadData(filepath)

    # get stop words from nltk
    stopWords = stopwords.words('english')

    # clean data 
    paragraphs = [cleanParagraph(para, stopWords) for para in paragraphs]
    paragraphs = [para.split('.') for para in paragraphs]

    # corpus [[w1,w2,w3..],[..]]
    corpus = []
    for i in tqdm(range(len(paragraphs))):
        for line in paragraphs[i]:
            words = [x for x in line.split()]
            corpus.append(words)
            num_of_sentences = len(corpus)

    num_of_words = 0
    for line in corpus:
        num_of_words += len(line)

    print('Num of sentences - %s'%(num_of_sentences))
    print('Num of words - %s'%(num_of_words))


    phrases = Phrases(sentences=corpus,min_count=25,threshold=50)
    bigram = Phraser(phrases)

    for index,sentence in enumerate(corpus):
        corpus[index] = bigram[sentence]


    #############################
    # Training process

    class callback(CallbackAny2Vec):
        '''Callback to print loss after each epoch.'''

        def __init__(self):
            self.epoch = 0

        def on_epoch_end(self, model):
            loss = model.get_latest_training_loss()
            if self.epoch == 0:
                print('Loss after epoch {}: {}'.format(self.epoch, loss))
            else:
                print('Loss after epoch {}: {}'.format(self.epoch, loss- self.loss_previous_step))
            self.epoch += 1
            self.loss_previous_step = loss
            
    # sg - skip gram |  window = size of the window | size = vector dimension
    size = 100
    window_size = 2 # sentences weren't too long, so
    epochs = 100
    min_count = 2
    workers = 4
        
    # train word2vec model using gensim
    model = Word2Vec(corpus, sg=1,window=window_size,size=size,
                    min_count=min_count,workers=workers,iter=epochs,sample=0.01, compute_loss=True)#, callbacks=[callback()])
                                    
    # save model
    model_filepath = filepath.replace("dataset", "models")
    model_filepath = model_filepath.replace(".txt", ".h5")

    model.save(model_filepath)
    model = Word2Vec.load(model_filepath)

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(
        description='WordEmbedding.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-dataset', type=str, required=True,
                       help="""Filepath to the dataset""")
    args = parser.parse_args()

    train(args.dataset)