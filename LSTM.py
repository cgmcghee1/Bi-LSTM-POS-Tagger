import nltk
import numpy as np
from nltk.corpus import treebank
import pickle
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

class Model:


    def __init__(self, corpus = treebank):


        tagged_words = np.asarray(corpus.tagged_words())
        self.tagset = {tag:num for (num,tag) in enumerate(set(tagged_words[:,1]))}
        self.indtag = {num:tag for (tag,num) in self.tagset.items()}
        self.word_index = {word:num for (num, word) in \
                           enumerate(set(tagged_words[:,0]))}
        self.word_index['UNK'] = len(self.word_index) + 1

        self.tagged_sents = corpus.tagged_sents()




    def train(self, save = False):



        self.X = np.zeros(shape=(len(self.tagged_sents), 100))
        self.y = np.zeros(shape=(len(self.tagged_sents), 100, len(self.tagset)))

        for index,sent in enumerate(self.tagged_sents):
            X_temp = []
            y_temp = []
            for (word,tag) in sent:
                X_temp.append(self.word_index[word])
                y_temp.append(self.tagset[tag])

            X_temp, y_temp = self.pad(X_temp,y_temp)
            y_temp = to_categorical(y_temp, num_classes = len(self.tagset))

            self.X[index,:] = np.asarray(X_temp).astype(np.float32)
            self.y[index,:,:] = np.asarray(y_temp).astype(np.float32)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split\
            (self.X, self.y, test_size = 0.1, random_state = 42)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split\
            (self.X_train, self.y_train, test_size = 0.1, random_state = 42)

        self.model = Sequential()
        self.model.add(Embedding(input_dim = len(self.word_index)+1, \
                            output_dim = 300,input_length = 100, trainable = True))
        self.model.add(Bidirectional(LSTM(len(self.tagset), return_sequences = True)))
        self.model.add(TimeDistributed(Dense(len(self.tagset), activation = 'softmax')))
        self.model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop',\
                      metrics =['accuracy'])

        self.model.fit(self.X_train, self.y_train, batch_size = 128, epochs = 10,\
                  validation_data = (self.X_val, self.y_val))
        if save == True:
            pickle.dump(self.model, open("model.p","wb"))
        return None

    def tagger(self, sentence):
        split_sent = nltk.word_tokenize(sentence)

        X = np.zeros(shape = (1,100))

        for index,word in enumerate(reversed(split_sent)):
            if word in self.word_index:
                X[0,99-index] = self.word_index[word]
            else:
                X[0,99-index] = self.word_index['UNK']

        if hasattr(self, "model"):
            tagsequence = self.model.predict(X)
        else:
            model = pickle.load(open("model.p","rb"))
            y = model.predict(X)

        tagsequence = []

        for i in y[0,:]:
            tag_pred = np.argmax(i)
            tagsequence.append(self.indtag[tag_pred])

        tagsequence = tagsequence[100-len(split_sent):]

        return tagsequence


    def pad(self,X,y):
        if len(X) < 100:
            for i in range(100-len(X)):
                X.insert(0,0)
                y.insert(0,0)
        elif len(X) > 100:
            del(X[100:])
            del(y[100:])
        return(X,y)
