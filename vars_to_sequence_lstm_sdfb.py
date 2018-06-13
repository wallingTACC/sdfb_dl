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
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import random as rand

from multiprocessing import Pool
import pickle

if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()

# Configure Tensorflow session
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
#sess = tf.Session(config=tf.ConfigProto(
#  allow_soft_placement=True, log_device_placement=True))

#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#sess = tf.Session(config=config)

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
#set_session(tf.Session(config=config))

# Set seeds to any randomness below
seed = 123
np.random.seed(seed)

PROCESS_DATA = False

if PROCESS_DATA:
    # Prep the data
    blocks = pd.read_csv('data/master_data_7_31_17_w_blocks.csv', low_memory=False)
    
    # Dev - Randomly select 1000
    blocks = blocks.iloc[rand.sample(range(len(blocks.index)), 1000)]
    
    # Save article ids for matching
    doc_ids = blocks.article_id
    
    # Pick columns of interest and drop missing
    blocks = blocks[['start_date', 'end_date', 'denom', 'occupation', 'gender', 'baptized', 'married', 'faith', 'Block']]
    
    # Clean up start_date and end_date
    blocks.start_date = pd.to_numeric(blocks.start_date, errors='coerce', downcast='integer')
    blocks.end_date = pd.to_numeric(blocks.end_date, errors='coerce', downcast='integer')
    
    #blocks = blocks.dropna()
    
    #X = blocks.loc[:, blocks.columns != 'Block']
    #y = blocks.loc[:,'Block']
    
    X = blocks
    dummy_X = pd.get_dummies(X, columns=['denom', 'occupation', 'gender', 'baptized', 'married', 'faith', 'Block'])
    #dummy_y = pd.get_dummies(y)
    
    num_vars = len(dummy_X.columns)
    #uniq_y = y.unique().size
    
    # Text Data
    # Need to read in a files in data dir matching article_ids
    article_text = []
    data_dir = 'data/ODNB_Entries_as_Textfiles/'
    for doc_id in doc_ids:
        file = data_dir + 'odnb_id_' + str(doc_id) + '.txt'
        print(file)
        text = open(file, 'r').read()
        if text != None:
            article_text.append(text)
        else:
            article_text.append('')


# Need to encapsulate this part in a function for use with multiprocessing
def row_encode(idx, row, words, seed_len=10, out_len=1, step=5):
    print(idx)
    new_data = pd.DataFrame()
    num_words = len(words)
    
    for j in range(0, num_words - seed_len, step):
        #row = data.iloc[idx]
        row['seed'] = words[j: j + seed_len]
        row['next_words'] = words[j+seed_len:j+seed_len+out_len]
        new_data = new_data.append(row) 
        
    return new_data

# Need to create 'seed' and 'next' text examples, where seed is length 3 and next length 1 word
# Each 'seed' becomes an observation for training
# Ex: 'The dog ran fast down the road'
#         'The dog ran' -> 'fast'
#         'dog ran fast' -> 'down'
#         'ran fast down' -> 'the'
def encode_text(data, text, seed_len=10, out_len=1, step=5):
    
    # First encode the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    encoded = tokenizer.texts_to_sequences(text)
    
    global dictionary # Save results for output
    dictionary = tokenizer.word_index
    
    p = Pool(24)
    new_data_list = p.starmap(row_encode, [(i, data.iloc[i], encoded[i]) for i in range(data.shape[0])])
    p.close()
    
    # Combine list of new dataframes to single one 
    result = pd.concat(new_data_list)
    
    return(result)
      
# Encoding stuff takes awhile, save it for re-use    
if PROCESS_DATA:
    encoded = encode_text(dummy_X, article_text)
    encoded.to_pickle('encoded.pkl')
    with open('dictionary.pkl', 'wb') as f:
        pickle.dump(dictionary, f)
else:
    encoded = pd.read_pickle('encoded.pkl')
    with open('dictionary.pkl', 'rb') as f:
        dictionary = pickle.load(f)

vocab_size = len(dictionary) + 1 # Column 0 is always 0 ?
X = encoded.loc[:, encoded.columns != 'next_words']
#X_text = encoded['seed']


seqs = [i for i in encoded['next_words']]
#y = k_utils.to_categorical(seqs, num_classes=vocab_size) # Stanard is to one-hot-encode the output
y = np.array(seqs) # one-hot encoding leads to out of memory errors, instead use straight integers with categorical_cross_entropy loss

# Save temp results
#with open('temp.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#    pickle.dump([X, y], f)
    
# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split X into text (seed) and normal vars
# Must put text into a matrix
X_text_mat_train = np.array([i for i in X_train['seed']])
X_vars_train = X_train.loc[:, X_train.columns != 'seed']

X_text_mat_test = np.array([i for i in X_test['seed']])
X_vars_test = X_test.loc[:, X_train.columns != 'seed']

num_vars = len(X_vars_train.columns) # Reset num_vars after above manipulations

# Build our model to predict words from variables
   
def lstm_w_vars():
    
    text_seed_len = 10
    
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
    

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])

    parallel_model = multi_gpu_model(model)
    parallel_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])
    
    return model, parallel_model

model, parallel_model = lstm_w_vars()

# checkpoint
filepath="weights-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=25)
callbacks_list = [checkpoint]

parallel_model.fit([X_text_mat_train, X_vars_train], y_train, epochs=500, batch_size=512, callbacks=callbacks_list)

model.set_weights(parallel_model.get_weights())

model.save('final_weights.h5')

#predict = model.predict([X_text_mat_test, X_vars_test])

# Each prediction gives us the  probability of the next word given previous words and data

#def generate_next_word(fitted_model, X_vars, seed=['the', 'dog', 'ran']):
    
#    seed_text_ints = np.asarray([dictionary[w] for w in seed]).reshape(1,len(seed))
#    probs = fitted_model.predict([seed_text_ints, X_vars])
#    
#    max_prob_idx = np.argmax(probs)
#    
#    words_idx = list(dictionary.keys())
#    
#    word = words_idx[max_prob_idx-1]
#    
#    return word

# Start generating text and keep adding to the result, sliding window 1 word each time
# Use the nth example in our test set
#test_idx = 5
#words_idx = list(dictionary.keys())
#test_seed = [words_idx[i-1] for i in X_text_mat_test[test_idx]]
#test_vars = X_vars_test.iloc[test_idx].reshape(1,10)
#result = test_seed
#for i in range(3):
#    seed = result[i:i+len(test_seed)]
#    next_word = generate_next_word(model, X, seed)
#    print(seed)
#    print(next_word)
#    result.append(next_word)
    
#print(result)
