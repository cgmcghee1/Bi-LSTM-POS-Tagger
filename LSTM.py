import nltk
import numpy as np
import pandas as pd
from nltk.corpus import treebank
from Functions import *
import pickle
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import InputLayer
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import TimeDistributed
from sklearn.model_selection import train_test_split


class Model:


    def __init__(self, corpus = treebank):

        """


        """
        self.embedding_dict = pickle.load(open("embeddings.p", "rb"))
        self.X = []
        self.y = []
        self.tagset = {tag:num for (num, tag) in\
                       enumerate(set([tag for (word,tag) in \
                                      corpus.tagged_words(tagset = 'universal')]))}

        for sent in corpus.tagged_sents(tagset = 'universal'):
            if len(sent) >= 100:
                continue
            X_temp = []
            y_temp = []
            for (word,tag) in sent:
                y_onehot = np.zeros(len(self.tagset))
                y_onehot[self.tagset[tag]] = 1
                y_temp.append(y_onehot)
                if word.lower() in self.embedding_dict:
                    X_temp.append(self.embedding_dict[word.lower()])
                else:
                    X_temp.append(np.random.uniform(-.1,.1,size = (300,)))

            X_temp, y_temp = pad(X_temp,y_temp,len(self.tagset))

            self.X.append(X_temp)
            self.y.append(y_temp)

        self.X = np.asarray(self.X).astype(np.float64)
        self.y = np.asarray(self.y).astype(np.float64)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split\
            (self.X, self.y, test_size = 0.1, random_state = 42)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split\
            (self.X_train, self.y_train, test_size = 0.1, random_state = 42)

    def train(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(len(self.tagset),input_shape =(100,300)\
                                     , return_sequences = True)))
        model.add(TimeDistributed(Dense(len(self.tagset), activation = 'softmax')))
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',\
                      metrics =['accuracy'])

        model.fit(self.X_train, self.y_train, batch_size = 128, epochs = 10,\
                  validation_data = (self.X_val, self.y_val))

        model.summary()
        return None

    def tagger(sentence, self):

        return tagsequence


a = Model().train()
b = 5

