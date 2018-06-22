#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 12:54:12 2018

This demo shows how to creat word generation models based on both text and categorical variables associated with that text.
It involves merging standard NN with LSTM based ones.

@author: walling
"""
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, SimpleRNN, Embedding, Activation, TimeDistributed, Dropout, Concatenate, concatenate
from keras.wrappers.scikit_learn import KerasClassifier
import keras.utils as k_utils

from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import os
        
# Execute process data to load variables.  NOTE: very unpythonic
from sdfb_process_data import *

# Configure Tensorflow session

#if 'session' in locals() and session is not None:
#    print('Close interactive session')
#    session.close()

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

# MODEL
vocab_size = len(dictionary) + 1 # Column 0 is always 0 ?
num_vars = len(dummy_X.columns)-1 # Drop doc_id when we pass to model iterations

# Data Generator
from random import shuffle

# Create train, val, test subsets

# NOTE: Not exact, i.e. num_samples != num_train+num_val+num_test
num_samples = len(seeds)
num_train = int(num_samples*.7)
num_val = int(num_samples*.1)
num_test = int(num_samples*.2)
indices = [i for i in range(num_samples)]
shuffle(indices)
train_idx = indices[:num_train]
val_idx = indices[(num_train):(num_train+num_val)]
test_idx = indices[(num_train+num_val):]

from sdfb_data_generator import SDFBDataGenerator
params = {'batch_size': 512,
          'shuffle': True}
training_generator = SDFBDataGenerator(train_idx, **params)
validation_generator = SDFBDataGenerator(val_idx, **params)

# Will remove doc_id

# Build our model to predict words from variables
   
def lstm_w_vars_sequential():
    
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
    model.add(Concatenate([lstm_model, var_model]))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))
    

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])

    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])
    
    return model, parallel_model

def lstm_w_vars_functional():
    
    text_seed_len = 10
    
    # LSTM 
    lstm_input = Input(shape=(text_seed_len,))
    lstm_model = Embedding(input_dim=vocab_size, output_dim=10, input_length=text_seed_len)(lstm_input)
    lstm_model = LSTM(units=10, return_sequences=True)(lstm_model)
    lstm_model = LSTM(units=10, return_sequences=False)(lstm_model)
    
    # Standard
    var_input = Input(shape=(num_vars,))
    var_model = Dense(units=64, activation='relu')(var_input)
    var_model = Dense(units=64, activation='relu')(var_model)
    
    # Merge and Output
    cat = concatenate([lstm_model, var_model])
    output = Dense(vocab_size, activation='softmax')(cat)
    
    model = Model(inputs=[lstm_input, var_input], outputs=[output])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])

    parallel_model = multi_gpu_model(model, gpus=4)
    parallel_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])
    
    return model, parallel_model

model, parallel_model = lstm_w_vars_functional()

# checkpoint
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # Not supported on $WORK
filepath="data/weights-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=1)
callbacks_list = [checkpoint]

#parallel_model.fit([X_text_mat_train, X_vars_train], y_train, epochs=500, batch_size=512, callbacks=callbacks_list)

parallel_model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    nb_epoch=100,
                    use_multiprocessing=False,
                    workers=16,
                    callbacks=callbacks_list)

# Get results back out
#model.set_weights(parallel_model.get_weights())

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
