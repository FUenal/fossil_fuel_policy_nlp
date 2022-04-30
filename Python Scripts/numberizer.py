#!usr/bin/env/python
"""
Authors: Fatih Uenal & Shashi Badloe
Description: Creates a conversion dictionary where every word in the training
data is mapped to an integer
Fourth step in the pipeline.
Source of the model: https://link.springer.com/article/10.1007%2Fs10113-020-01677-8
"""
import sqlite3
from functools import reduce
import os

def connect_to_db(db_name):
    """

    db_name: string, name of the database to connect to
    """
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    return c,conn
    
def collect_labeled_data(c):
    """Returns list with all words from all blocks

    c: database cursor
    """
    c.execute('SELECT block FROM {}'.format('Labeled_data'))
    all_blocks = c.fetchall()
    all_blocks = list(map(lambda tup: eval("["+tup[0]+"]" ), all_blocks))
    all_blocks = reduce(lambda lis1,lis2: lis1+lis2,all_blocks)
    
    return all_blocks

def word_to_number(words):
    """Returns a dictionary where every unique word is coupled to a number

    words: list of strings, all words from all blocks
    """
    conv_dict = {}
    conv_dict["<PAD>"] = 0
    conv_dict["<START>"] = 1
    conv_dict["<UNK>"] = 2  # unknown
    conv_dict["<UNUSED>"] = 3

    i = 4
    for word in words:
        if word not in conv_dict.keys():
            conv_dict[word] = i
            i += 1
    return conv_dict
    
def save_dict(filename,dictionary):
    """Writes dictionary to a textfile

    filename: string, name of the resulting text file
    dictionary: dict, any dictionary
    """
    txt_file = open(filename,"w", encoding = "UTF-8")
    txt_file.write(str(dictionary))
    txt_file.close()
    
def reset_dict(filename):
    """Removes conversion dictionary if it already exists"""
    if os.path.isfile(filename):
        os.remove(filename)
    
if __name__ == '__main__':
    dict_name = "conversion_dictionary.txt"
    reset_dict(dict_name)
    c,conn = connect_to_db('climate.db')
    words = collect_labeled_data(c)
    conn.commit()
    conv_dict = word_to_number(words)
    save_dict(dict_name,conv_dict)
