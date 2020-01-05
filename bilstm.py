import sys
sys.path.append('../')

import os
import numpy as np
import pandas as pd
from data import *
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.recurrent import GRU
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.layers.core import Dropout
import nltk
from nltk.tokenize import TweetTokenizer



def bilstm(layer_data, act_func, embed_dim, max_feat, opti, epochs, process):
    dense_layer_act = ["softmax", "sigmoid", "tanh", "relu"]
    optimizer = ["sgd", "nadam", "adam", "adamax", "adadelta", "rmsprop"]
    
    embed_dim =int(embed_dim)
    max_feat = int(max_feat)
    epochs = int(epochs)
    layer_data = layer_data.split(",")
    num_of_layers = int(layer_data[0])
    print(layer_data)
    print(num_of_layers)
    if num_of_layers != len(layer_data) - 1:
        print("genetic error : layer_data incorrect")
    layer_data = layer_data[1:]
    for i in range(len(layer_data)):
        layer_data[i] = int(layer_data[i])
    act_func = dense_layer_act[int(act_func)]
    curr_opti = optimizer[int(opti)]

    print("num_of_layers : " + str(num_of_layers))
    print("layer info : " + str(layer_data))
    print("act_func : " + str(act_func))
    print("opti : " + str(curr_opti))
    print("max_feat : " + str(max_feat))
    print("embed_dim : " + str(embed_dim))

    sw = load_sw()
    data = load_data()

    tt = TweetTokenizer()
    stemmer = nltk.stem.SnowballStemmer('english')

    X = []
    Y = []

    for i, k in enumerate(data.keys()):
        for text in data[k]:
            text = line_clean(text)
            text = line_sw(text, tt, sw)
            text = line_stem(text, tt, stemmer)
            X.append(text)
            Y.append(i)

    data = pd.DataFrame(
            {
                'text': X,
                'label': Y
                })

    max_features = max_feat
    tokenizer = Tokenizer(num_words=max_features, split=' ')
    tokenizer.fit_on_texts(data['text'].values)
    X = tokenizer.texts_to_sequences(data['text'].values)
    X = pad_sequences(X)

    embed_dim = embed_dim
    lstm_out = 128

    model = Sequential()
    model.add(Embedding(max_features, embed_dim,input_length = X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    
    for i in range(len(layer_data) - 1):
        model.add(LSTM(layer_data[i], dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(LSTM(layer_data[-1], dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(8,activation=act_func))
    model.compile(loss = 'categorical_crossentropy', optimizer=curr_opti ,metrics = ['accuracy'])
    print(model.summary())

    Y = pd.get_dummies(data['label']).values
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 5)
    print(X_train.shape,Y_train.shape)
    print(X_test.shape,Y_test.shape)

    batch_size = 128
    history = model.fit(X_train, Y_train, epochs = epochs, batch_size=batch_size, verbose = 2, validation_split=0.2)

    score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)

    train_g = history.history['loss']
    val_g = history.history['val_loss']
    iter_epoch = epochs

    for i in range(len(train_g)):
        if train_g[i] > val_g[i]:
            iter_epoch = i + 1

    if int(process) == 0:
        return iter_epoch
    else:
        return acc
