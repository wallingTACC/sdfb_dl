# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Set seeds to any randomness below
seed = 123
np.random.seed(seed)

# Prep the data

blocks = pd.read_csv('master_data_7_31_17_w_blocks.csv', low_memory=False)

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

def quick_test():
    model = get_model_simple()
    X_train, X_test, y_train, y_test = train_test_split(dummy_X, dummy_y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train, epochs=5, batch_size=32)
    
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
    
if __name__ == "__main__":
    quick_test()
    #full_validation()

