"""
__info: This file is to convert Vietnamese summarization raw data into tokenized chunks for summarization model.
        This file is modified and mostly borrowed code from https://github.com/becxer/cnn-dailymail/
        Added some modifications to adapt to Vietnamese dataset (https://github.com/ThanhChinhBK/vietnews/blob/master/data/)
__original-author__ = "Abigail See" + converted to python3 by Becxer
__modified-author__ = "Vy Thai"
__email__ = "vythai@stanford.edu"
"""

import tensorflow as tf
from tensorflow.core.example import example_pb2
import sys
import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2
from os import listdir
import collections
import pandas as pd
import collections
from collections import Counter
import random
from itertools import islice

finished_files_dir = "finished_files"

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
VOCAB_SIZE = 200000
CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data
dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence
chunks_dir = os.path.join(finished_files_dir, "chunked")


def read_data_csv(path):
    data_orig = pd.read_csv(path, header=0)
    data = list(data_orig.file_name)
    list_ = [1290, 281, 717, 4772, 1982, 705, 2524]
    #list_  = {4: 4772, 7: 2524, 5: 1982, 1: 1290, 3: 717, 6: 705, 2: 281})

    assert(len(data) == 12271)
    temp = iter(data)
    res = [list(islice(temp, 0, ele)) for ele in list_]
    assert(len(res) == 7)

    list_1 = res[0]
    list_2 = res[1]
    list_3 = res[2]
    list_4 = res[3]
    list_5 = res[4]
    list_6 = res[5]
    list_7 = res[6]

    chosen_1 =  random.sample(list_1, 164)
    chosen_2 =  random.sample(list_2, 37)
    chosen_3 =  random.sample(list_3, 80)
    chosen_4 =  random.sample(list_4, 592)
    chosen_5 =  random.sample(list_5, 239)
    chosen_6 =  random.sample(list_6, 81)
    chosen_7 =  random.sample(list_7, 340)

    total_change = chosen_1 + chosen_2 + chosen_3 + chosen_4 + chosen_5 + chosen_6 + chosen_7
    assert(len(total_change) == len(set(total_change)))

    #change items in csv files
    for i in total_change:
        data_orig['file_name'] = data_orig['file_name'].replace({i: "val_"+i})
        os.rename(directory+i, directory+"val_" +i)

    data_orig.to_csv("basic/EmoLabel/NEW_updated_train_val.csv", index=False)



    """
    size_list = len(data)
    length_counts = Counter(word for word in data)
    print(length_counts)

    data = set(data)
    size_set = len(data)
    print('Size list labels: ', size_list, "Size set labels: ", size_set)
    #assert(size_list == size_set)
    """



    return data
# This function is to take dir and load data and process them.
def read_images_name(directory):
    documents = listdir(directory)
    size_list = len(documents)

    documents = set(documents)
    size_set = len(documents)
    print('Size list pics: ', size_list, "Size set pics: ", size_set)

    #assert(size_list == size_set)

    return documents

def remove(dir):
    for i in dir:
        os.remove(directory + i)


directory = 'basic/Image/aligned/'
path = 'basic/label/train_label.csv'
#print(stories[1]['highlights'])
def testing(path, dir):
    data_orig = pd.read_csv(path, header=0)
    data = list(data_orig.label)
    size_list = len(data)
    length_counts = Counter(word for word in data)
    print(length_counts)

    data = set(data)
    size_set = len(data)
    print('Size list labels: ', size_list, "Size set labels: ", size_set)
    # assert(size_list == size_set)
if __name__ == '__main__':
    #labels = read_data_csv(path)
    testing(path, directory)
    #images = read_images_name(directory)

    #intersec = labels & images
    #print(len(intersec))
    #labels_diff = labels - intersec
    #images_diff = images - intersec

    #print("Different labels: ", labels_diff)
    #print("Different images: ", images_diff)
    #remove(images_diff)

