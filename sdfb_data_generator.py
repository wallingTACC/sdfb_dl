# -*- coding: utf-8 -*-
# Ref: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
import numpy as np
import pandas as pd
import pickle
import keras


class SDFBDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        
        # Preload the data
        with open('data/doc_ids.pkl', 'rb') as f:
            self.doc_ids = pickle.load(f)
        with open('data/dictionary.pkl', 'rb') as f:
            self.dictionary = pickle.load(f)
        with open('data/article_text.pkl', 'rb') as f:
            self.article_text = pickle.load(f)
        self.dummy_X = pd.read_pickle('data/dummy_X.pkl')
        self.dummy_X.set_index('doc_id')
        self.seeds = pd.read_pickle('data/seeds.pkl')
        self.dummy_X.set_index('doc_id')

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X_text_mat_train, dummy_X_temp, y_temp = self.__data_generation(list_IDs_temp)

        # Must pass in [X1, X2], y format for multi-input models
        return [X_text_mat_train, dummy_X_temp], y_temp

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        
        seeds_temp = self.seeds.iloc[list_IDs_temp]
        doc_ids_temp = seeds_temp.id
        
        dummy_X_temp = self.dummy_X[self.dummy_X['doc_id'].isin(doc_ids_temp)]
        
        # Join seeds_temp to dummy_X_temp on doc_id=id
        joined_temp = dummy_X_temp.join(seeds_temp, on='doc_id')
        
        # Drop doc_id column
        dummy_X_temp = dummy_X_temp.drop('doc_id', axis=1)
        
        X_text_mat_train = np.array([i for i in seeds_temp['seed']])
        y_temp = np.array([i for i in seeds_temp['next_words']])
        
        return X_text_mat_train, dummy_X_temp, y_temp
    
#        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
#        # Initialization
#        X = np.empty((self.batch_size, *self.dim, self.n_channels))
#        y = np.empty((self.batch_size), dtype=int)
#
#        # Generate data
#        for i, ID in enumerate(list_IDs_temp):
#            # Store sample
#            X[i,] = np.load('data/' + ID + '.npy')
#
#            # Store class
#            y[i] = self.labels[ID]
#
#        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
