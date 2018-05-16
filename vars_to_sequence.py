#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 12:54:12 2018

This demo shows how to create word predictions based on categorical variables.

The dummy dataset makes it easier to see how variables (X) should match up to words (Y_text).

@author: walling
"""
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, Embedding, Activation, TimeDistributed, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import random as rand

# Set seed for reproducible results
seed = 12
np.random.seed(seed)

X1 = ['dog', 'dog', 'cat', 'cat', 'apple', 'apple'] * 1000
X2 = ['ran', 'walked', 'ran', 'walked', 'are', 'taste'] * 1000
X3 = ['fast', 'fast', 'slow', 'slow', 'good', 'good'] * 1000

X = pd.DataFrame(list(zip(X1, X2, X3)))

Y_text = ['Dog ran fast', 'Dog walked fast', 'Cat ran slow', 'Cat walked slow', 'Apples are good', 'Apples taste good'] * 1000

# One-hot encode categorical variables
dummy_X = pd.get_dummies(X)

# Need to convert text to matrix of indexes
def word_tokenizer(text):
    # only work with the 10 most popular words found in our dataset
    max_words = 10
    
    # create a new Tokenizer
    tokenizer = Tokenizer(num_words=max_words)
    
    # feed our text to the Tokenizer
    tokenizer.fit_on_texts(text)
    
    # Tokenizers come with a convenient list of words and IDs
    global dictionary # Save results for output
    dictionary = tokenizer.word_index
    
    def convert_text_to_index_array(text):
        # one really important thing that `text_to_word_sequence` does
        # is make all texts the same length -- in this case, the length
        # of the longest text in the set.
        return [dictionary[word] for word in kpt.text_to_word_sequence(text)]
    
    allWordIndices = []
    # for each text, change each token to its ID in the Tokenizer's word_index
    for t in text:
        wordIndices = convert_text_to_index_array(t)
        allWordIndices.append(wordIndices)
    
    # now we have a list of all text converted to index arrays.
    # cast as an array for future usage.
    #allWordIndices = np.asarray(allWordIndices)

    # create one-hot matrices out of the indexed tweets
    result = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
    
    return result

y = word_tokenizer(Y_text)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(dummy_X, y, test_size=0.2, random_state=42)

# Build our model to predict words from variables
def simple():
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=10)) # 10 dummy variables
    model.add(Dense(10)) # 10 possible words
    #model.add(Dropout(0.3)) # Prevent over fitting
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    
    return model

def deeper():
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=10)) # 10 dummary variables
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(10)) # 10 possible words
    #model.add(Dropout(0.3)) # Prevent over fitting
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    
    return model
    
def lstm():
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=3)) # 10 variables
    model.add(Dense(10)) # 10 possible words
    #model.add(Dropout(0.3)) # Prevent over fitting
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    
    return model

model = deeper()
model.fit(X_train, y_train, epochs=30, batch_size=100)

# Make Precictions
predict = model.predict(X_test)

# Transform back to words
words_idx = list(dictionary.keys())

# Print results for first N test sets
for i in range(10):
    vari = X_test.iloc[i]
    rowi_probs = predict[i,:]
    maxi_idx = np.argpartition(rowi_probs, -3)[-3:] # Top 3 max probs indexes
    texti = [words_idx[i] for i in maxi_idx]
    
    # Show results
    print('--------')
    print(vari)
    print(texti)



