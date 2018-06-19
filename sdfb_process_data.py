# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import random as rand
from multiprocessing import Pool
import pickle

import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

def get_data():
    # Prep the data
    blocks = pd.read_csv('data/master_data_7_31_17_w_blocks.csv', low_memory=False)
    
    # Dev - Randomly select 1000
    blocks = blocks.iloc[rand.sample(range(len(blocks.index)), 1000)]
   
    # Don't include doc_ids in independent vars, but need for processing article_text
    doc_ids = blocks.article_id
    
    # Pick columns of interest and drop missing
    blocks = blocks[['start_date', 'end_date', 'denom', 'occupation', 'gender', 'baptized', 'married', 'faith', 'Block']]
    
    # Clean up start_date and end_date
    blocks.start_date = pd.to_numeric(blocks.start_date, errors='coerce', downcast='integer')
    blocks.end_date = pd.to_numeric(blocks.end_date, errors='coerce', downcast='integer')
    
    #blocks = blocks.dropna()

    X = blocks
    dummy_X = pd.get_dummies(X, columns=['denom', 'occupation', 'gender', 'baptized', 'married', 'faith', 'Block'])
    dummy_X['doc_id'] = doc_ids
    
    with open('data/doc_ids.pkl', 'wb') as f:
        pickle.dump(doc_ids, f)
    dummy_X.to_pickle('data/dummy_X.pkl')
    
    return(doc_ids, dummy_X)

def get_articles(doc_ids):
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

    with open('article_text.pkl', 'wb') as f:
        pickle.dump(article_text, f)
        
    return article_text

# Need to encapsulate this part in a function for use with multiprocessing
def row_encode(doc_id, words, seed_len=10, out_len=1, step=5):

    print(doc_id)
    num_words = len(words)
    
    row_list = []    
    for j in range(0, num_words - seed_len, step):
        #row = data.iloc[idx]
        row = {}
        row['doc_id'] = doc_id
        row['seed'] = words[j: j + seed_len]
        row['next_words'] = words[j+seed_len:j+seed_len+out_len]
        row_list.append(row) 

    # Checkpoint
    new_data = pd.DataFrame(row_list)
    #new_data.to_pickle('data/pickle/'+str(idx)+'.pkl')
        
    return new_data

# Need to create 'seed' and 'next' text examples, where seed is length 3 and next length 1 word
# Each 'seed' becomes an observation for training
# Ex: 'The dog ran fast down the road'
#         'The dog ran' -> 'fast'
#         'dog ran fast' -> 'down'
#         'ran fast down' -> 'the'
def encode_text(doc_ids, text, seed_len=10, out_len=1, step=5):
    
    # First encode the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    encoded = tokenizer.texts_to_sequences(text)
    
    global dictionary # Save results for output
    dictionary = tokenizer.word_index
    
    p = Pool(6)
    new_data_list = p.starmap(row_encode, [(doc_ids.iloc[i], encoded[i]) for i in range(len(doc_ids))])
    p.close()
    
    # Combine list of new dataframes to single one 
    result = pd.concat(new_data_list)
   
    result.to_pickle('data/seeds.pkl')
    with open('data/dictionary.pkl', 'wb') as f:
        pickle.dump(dictionary, f)
 
    return(result)



PROCESS_DATA = False 

# Encoding stuff takes awhile, save it for re-use 
if PROCESS_DATA:
    seed = 123
    rand.seed(seed)
    doc_ids, dummy_X = get_data()
    article_text = get_articles(doc_ids)
    seeds = encode_text(doc_ids, article_text)
    
else:
    with open('data/doc_ids.pkl', 'rb') as f:
        doc_ids = pickle.load(f)
    with open('data/dictionary.pkl', 'rb') as f:
        dictionary = pickle.load(f)
    with open('data/article_text.pkl', 'rb') as f:
        article_text = pickle.load(f)
    dummy_X = pd.read_pickle('data/dummy_X.pkl')
    seeds = pd.read_pickle('data/seeds.pkl')
