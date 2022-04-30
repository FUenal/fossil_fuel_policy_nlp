#!usr/bin/env/python
"""
Authors: Fatih Uenal & Shashi Badloe
Description: Prediction on unlabeled data using the model that was built
previously
Source of the model: https://link.springer.com/article/10.1007%2Fs10113-020-01677-8
"""
import tensorflow as tf
from numberizer import connect_to_db
from tensorflow import keras
import numpy as np
import sqlite3
import os
import pandas
import time

def load_conv_dict(filename):
    """Returns conversion dictionary for words to numbers

    filename: string, name of the dictionary textfile
    """
    conv_dict = open(filename, "r", encoding = 'UTF-8').read()
    return eval(conv_dict)

def load_model(model_directory):
    """Returns tensorflow Estimator object containing trained tensors

    model_directory: string, directory of the trained Saved Estimator
    """
    full_model_dir = os.path.join(model_directory,sorted(os.listdir(model_directory))[0])
    print('Loading model from "{}"'.format(full_model_dir))
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],\
                                   full_model_dir)
        predictor = tf.contrib.predictor.from_saved_model(full_model_dir)
        return predictor
    
def load_metadata(model_directory):
    """Returns integer of required tensor input data size

    model_directory: string, path where the metadata.txt is stored
    """
    filename = 'metadata.txt'
    metadata = open(os.path.join(model_directory,filename), mode = 'r')\
               .readline()
    return int(metadata)

def text_to_num(block, conv_dict):
    """Converts the words to numbers

    blocks: list of lists, every list contains all words of a block
    conv_dict: dictionary, conversion dictionary from text to numbers

    Because this is used for new data, there is a chance that some words
    are not in the conv_dict. These will be mapped to the number 2 which is
    marked as <UNK> unknown
    """
    def convert_word(conv_dict, word):
        try:
            return conv_dict[word]
        except:
            return 2
        
    words = list(map(lambda word: convert_word(conv_dict,word), block))
    return words

def insert_padding(block,conv_dict):
    """Returns blocks of same length where text is converted to numbers

    blocks: list of lists, every list contains all words of a block
    conv_dict: dictionary, conversion dictionary from text to numbers

    padding of 0's is added at the end to make them the same length
    """
    #Find length of the longest block and add padding
    block = keras.preprocessing.sequence.pad_sequences([block],\
                                                value=conv_dict["<UNUSED>"],\
                                                padding='post',\
                                                maxlen=load_metadata(model_directory))
    block = block[0]
    return np.array(block)

def lables_to_text(result):
    """Returns string with name of the label

    result: tuple of int(label),list(probabilities)
    """
    if result[0] == 0:
        label = 'Adaptation'
    elif result[0] == 1:
        label = 'Mitigation'
    elif result[0] == 2:
        label = 'Non-climate'
    return (label,result[1])

def collect_data_unlabeled(c,table_name):
    """Returns list with text blocks

    c: database cursor
    table_name: string, name of the table containing unlabeled data

    """
    c.execute('SELECT COUNT(*) FROM {}'.format(table_name))
    len_data = int(c.fetchall()[0][0])
    print('{} rows in {}'.format(len_data,table_name))
    for i in range(1,len_data+1):
        c.execute('SELECT block FROM {} WHERE id = {}'.format(table_name,i))
        data = c.fetchall()
        block = eval("["+data[0][0]+"]")
        yield i,block

def predict(classifier,unlabeled_data):
    """Returns the predicted class and probabilties of all classes

    classifier: Estimator object obtained from load_model
    unlabeled_data: list of integers, numberized contents of a single block
    """
    model_input = tf.train.Example(features=tf.train.Features(\
    feature={"words": tf.train.Feature(int64_list=tf.train.Int64List\
                                         (value=unlabeled_data)) }))
    model_input = model_input.SerializeToString()
    output_dict = classifier({"predictor_inputs":[model_input]})
    y_probability = output_dict['probabilities'][0]
    y_predicted = output_dict["pred_output_classes"][0]
    return y_predicted,y_probability

def predict_blocks(classifier,block):
    """Returns tuple with predicted label and probabilities

    classifier: Estimator object obtained from load_model
    blocks: np array list of lists, contains all numberized blocks
    """
    result = predict(classifier,block)
    result = lables_to_text(result)
    return result

def add_column(table_entity,cursor,col_name_type):
    """Adds a column with col_name to database

    table_entity: string, name of the table
    cursor: database cursor
    col_name: tuple of strings (column name,column datatype)
    """
    try:
        cursor.execute("""ALTER TABLE {} ADD {} {}"""\
                       .format(table_entity,col_name_type[0],col_name_type[1]))
    except sqlite3.OperationalError:
        pass
        
def update_predictions_db(table_entity,cursor,predictions,i):
    """Updates the Unlabeled_data in the database with predicted labels"""
    parameters = (predictions[0],str(list(predictions[1]))[1:-1],i)
    cursor.execute("""UPDATE {} SET pred_class = ?, probabilities = ? WHERE id = ?"""\
                    .format(table_entity),parameters)

        
def create_metadata(filename,conv_dict, classifier):
    """Transforms vocabulary with metadata into TSV for use in tensorboard"""
    metadata = open(filename, 'w')
    reverse_dict = {value:key for key,value in conv_dict.items()}
    blocks = list(map(lambda num: [num], sorted(reverse_dict.keys())))
    blocks = list(map(lambda num: insert_padding(num,conv_dict),blocks))
    results = list(map(lambda num: predict_blocks(classifier,num),blocks))
    metadata.write('index\tword\tAdaptation\tMitigation\tNon-climate\n')
    for i,key in enumerate(sorted(reverse_dict.keys())):
        metadata.write('\t'.join([str(key),reverse_dict[key],\
                                  str(results[i][1][0]),\
                                  str(results[i][1][1]),\
                                  str(results[i][1][2])])+'\n')
    metadata.close()
    

    
    
if __name__ == '__main__':
    model_directory = os.path.join('..','tensorflow','logdir')
    start_time = time.perf_counter()
    classifier = load_model(model_directory)
    print("Loading conv_dict")
    conv_dict = load_conv_dict("conversion_dictionary.txt")
    print("Connecting to db")
    c,conn = connect_to_db("climate.db")
    add_column('Unlabeled_data',c,('probabilities','TEXT'))
    print("Retrieving blocks")
    for i,block in collect_data_unlabeled(c,'Unlabeled_data'):
        block = text_to_num(block,conv_dict)
        block = insert_padding(block,conv_dict)
        predictions = predict_blocks(classifier,block)
        update_predictions_db('Unlabeled_data',c,predictions,i)
        if i % 100000 == 0:
            print(i)
            conn.commit()
    conn.commit()
    print('Took {} seconds to update predictions'.format(int(time.perf_counter() - start_time)))
    print("Creating tensorboard metadata")
    create_metadata('tensorboard_metadata.txt',conv_dict, classifier)
    print("Predictions finished!")
