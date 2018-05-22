#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 12:54:12 2018

This demo shows how to creat word generation models based on both text and categorical variables associated with that text.
It involves merging standard NN with LSTM based ones.

@author: walling
"""
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, Embedding, Activation, TimeDistributed, Dropout, Merge
from keras.wrappers.scikit_learn import KerasClassifier
import keras.utils as k_utils
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

X1 = ['dog', 'dog', 'cat', 'cat', 'apple', 'apple'] * 10
X2 = ['ran', 'walked', 'ran', 'walked', 'are', 'taste'] * 10
X3 = ['fast', 'fast', 'slow', 'slow', 'good', 'good'] * 10

X = pd.DataFrame(list(zip(X1, X2, X3)))

Y_text = ['The dog ran fast down the road', 'The dog walked fast along the trail', 
          'A cat ran slow up the fence', 'The cat walked slow on the branch', 
          'Apples are good for your health', 'Apples taste good and clean your teeth'] * 10
# One-hot encode categorical variables
dummy_X = pd.get_dummies(X)

# Need to create 'seed' and 'next' text examples, where seed is length 3 and next length 1 word
# Each 'seed' becomes an observation for training
# Ex: 'The dog ran fast down the road'
#         'The dog ran' -> 'fast'
#         'dog ran fast' -> 'down'
#         'ran fast down' -> 'the'
def encode_text(data, text, seed_len=4, out_len=1, step=1):
    
    # First encode the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    encoded = tokenizer.texts_to_sequences(text)
    
    global dictionary # Save results for output
    dictionary = tokenizer.word_index
    
    # Want to match up categorical vars and the encoded text
    new_data = pd.DataFrame()
    for i in range(data.shape[0]): # Num rows
        words = encoded[i]
        num_words = len(words)
        
        for j in range(0, num_words - seed_len, step):
            row = data.iloc[i]
            row['seed'] = words[j: j + seed_len]
            row['next_words'] = words[j+seed_len:j+seed_len+out_len]
            new_data = new_data.append(row)
            
    return new_data        


encoded = encode_text(dummy_X, Y_text)

vocab_size = len(dictionary) + 1
X = encoded.loc[:, encoded.columns != 'next_words']
#X_text = encoded['seed']

# Have to one-hot-encode the output
seqs = [i for i in encoded['next_words']]
y = k_utils.to_categorical(seqs, num_classes=vocab_size)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split X into text (seed) and normal vars
X_text_mat_train = np.array([i for i in X_train['seed']])
X_vars_train = X_train.loc[:, X_train.columns != 'seed']

X_text_mat_test = np.array([i for i in X_test['seed']])
X_vars_test = X_test.loc[:, X_train.columns != 'seed']

# Build our model to predict words from variables
   
def lstm_w_vars():
    
    text_seed_len = 4
    num_vars = 10
    
    # LSTM 
    lstm_model = Sequential()
    lstm_model.add(Embedding(input_dim=vocab_size, output_dim=10, input_length=text_seed_len)) # Output_dim similar to units, is tunable
    lstm_model.add(LSTM(units=10, return_sequences=True))
    lstm_model.add(LSTM(units=10, return_sequences=False))
    
    # Standard
    var_model = Sequential()
    var_model.add(Dense(units=64, activation='relu', input_dim=num_vars))
    var_model.add(Dense(units=64, activation='relu'))
    
    # Merge
    model = Sequential()
    model.add(Merge([lstm_model, var_model], mode = 'concat'))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))
    

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    
    return model

model = lstm_w_vars()


model.fit([X_text_mat_train, X_vars_train], y_train, epochs=100, batch_size=10)

predict = model.predict([X_text_mat_test, X_vars_test])

# Each prediction gives us the  probability of the next word given previous words and data

def generate_next_word(fitted_model, X_vars, seed=['the', 'dog', 'ran']):
    
    seed_text_ints = np.asarray([dictionary[w] for w in seed]).reshape(1,4)
    probs = fitted_model.predict([seed_text_ints, X_vars])
    
    max_prob_idx = np.argmax(probs)
    words_idx = list(dictionary.keys())
    
    word = words_idx[max_prob_idx-1]
    
    return word

# Start generating text and keep adding to the result, sliding window 1 word each time
# Use the nth example in our test set
test_idx = 5
words_idx = list(dictionary.keys())
test_seed = [words_idx[i-1] for i in X_text_mat_test[test_idx]]
test_vars = X_vars_test.iloc[test_idx].reshape(1,10)
result = test_seed
for i in range(3):
    seed = result[i:i+4]
    next_word = generate_next_word(model, X, seed)
    print(seed)
    print(next_word)
    result.append(next_word)
    
print(result)
