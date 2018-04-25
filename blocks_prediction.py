# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import random as rand
import glob

# Set seeds to any randomness below
seed = 123
np.random.seed(seed)

# Prep the data

blocks = pd.read_csv('data/master_data_7_31_17_w_blocks.csv', low_memory=False)

# Dev - Randomly select 1000
#blocks = blocks.iloc[rand.sample(range(len(blocks.index)), 1000)]

# Save article ids for matching
doc_ids = blocks.article_id

# Pick columns of interest and drop missing
blocks = blocks[['start_date', 'end_date', 'denom', 'occupation', 'gender', 'baptized', 'married', 'faith', 'Block']]

# Clean up start_date and end_date
blocks.start_date = pd.to_numeric(blocks.start_date, errors='coerce', downcast='integer')
blocks.end_date = pd.to_numeric(blocks.end_date, errors='coerce', downcast='integer')

#blocks = blocks.dropna()

X = blocks.loc[:, blocks.columns != 'Block']
y = blocks.loc[:,'Block']

dummy_X = pd.get_dummies(X, columns=['denom', 'occupation', 'gender', 'baptized', 'married', 'faith'])
dummy_y = pd.get_dummies(y)

num_vars = len(dummy_X.columns)
uniq_y = y.unique().size

# Text Data
# Need to read in a files in data dir matching article_ids
article_text = []
data_dir = '/data/sdfb/ODNB_Entries_as_Textfiles/'
for doc_id in doc_ids:
    file = data_dir + 'odnb_id_' + str(doc_id) + '.txt'
    print(file)
    text = open(file, 'r').read()
    if text != None:
        article_text.append(text)
    else:
        article_text.append('')


# Build model
# Must define as a function to work with KerasClassifier
def get_model_simple():
    model = Sequential()
    
    model.add(Dense(units=64, activation='relu', input_dim=num_vars))
    model.add(Dense(units=uniq_y, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model

def get_model_complex():
    model = Sequential()
    
    model.add(Dense(units=64, activation='relu', input_dim=num_vars))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=uniq_y, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model

def get_model_text():
    model = Sequential()
    
    #model.add(Dense(units=64, activation='relu', input_dim=1))
    
    model.add(Dense(units=64, activation='tanh', input_shape=(3000,)))
    
    model.add(Dense(units=uniq_y, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model

def word_tokenizer(X):
    # only work with the 3000 most popular words found in our dataset
    max_words = 3000
    
    # create a new Tokenizer
    tokenizer = Tokenizer(num_words=max_words)
    
    # feed our odnb articles to the Tokenizer
    tokenizer.fit_on_texts(X)
    
    # Tokenizers come with a convenient list of words and IDs
    dictionary = tokenizer.word_index
    
    # Let's save this out so we can use it later
    #with open('dictionary.json', 'w') as dictionary_file:
    #    json.dump(dictionary, dictionary_file)
    
    def convert_text_to_index_array(text):
        # one really important thing that `text_to_word_sequence` does
        # is make all texts the same length -- in this case, the length
        # of the longest text in the set.
        return [dictionary[word] for word in kpt.text_to_word_sequence(text)]
    
    allWordIndices = []
    # for each tweet, change each token to its ID in the Tokenizer's word_index
    for text in X:
        wordIndices = convert_text_to_index_array(text)
        allWordIndices.append(wordIndices)
    
    # now we have a list of all text converted to index arrays.
    # cast as an array for future usage.
    allWordIndices = np.asarray(allWordIndices)

    # create one-hot matrices out of the indexed tweets
    X = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
    
    return X
    
def text_test():
    model = get_model_text()
    X = word_tokenizer(article_text)
    X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train, epochs=20, batch_size=32)
    
def quick_test():
    model = get_model_simple()
    X_train, X_test, y_train, y_test = train_test_split(dummy_X, dummy_y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train, epochs=20, batch_size=32)
    
    # TODO Add test validation steps

def full_validation():
    # Helper function for leveraging sklearn functionality for testing and validation
    estimator = KerasClassifier(build_fn=get_model_complex, epochs=5, 
                                batch_size=32, 
                                verbose=1)
    
    # This will train n_splits number of models
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    
    results = cross_val_score(estimator, dummy_X, dummy_y, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

def full_validation_text():
    # Helper function for leveraging sklearn functionality for testing and validation
    estimator = KerasClassifier(build_fn=get_model_text, epochs=5, 
                                batch_size=32, 
                                verbose=1)
    
    # This will train n_splits number of models
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    
    X = word_tokenizer(article_text)
    
    results = cross_val_score(estimator, X, dummy_y, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    
if __name__ == "__main__":
    full_validation_text()
    #full_validation()

