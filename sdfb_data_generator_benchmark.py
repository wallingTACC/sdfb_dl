#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 09:02:09 2018

@author: walling
"""
from sdfb_process_data import *

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


import timeit

timeit.timeit('training_generator.__getitem__()', number=1000)