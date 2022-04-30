#!usr/bin/env/python
"""
Authors: Fatih Uenal & Shashi Badloe
Description: TF topic classification using Bag of Words model
Fifth step in the pipeline.
Source of the model: https://link.springer.com/article/10.1007%2Fs10113-020-01677-8
Source of the model: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/learn/iris_custom_model.py
"""

import tensorflow as tf
from tensorflow import keras
from sklearn import metrics
import shutil
import pandas
import numpy as np
import sqlite3
import os
from numberizer import connect_to_db
import time
import datetime
import re

def load_conv_dict(filename):
    """Returns conversion dictionary for words to numbers

    filename: string, name of the dictionary textfile
    """
    conv_dict = open(filename, "r", encoding = 'UTF-8').read()
    return eval(conv_dict)

def collect_data_label(c,table_name):
    """Returns list with text blocks and list with block labels

    c: database cursor

    both lists must be of equal length
    """
    c.execute('SELECT block,class,source FROM {}'.format(table_name))
    data = c.fetchall()
    blocks = list(map(lambda tup: eval("["+tup[0]+"]" ), data))
    labels = list(map(lambda tup: eval("['"+tup[1]+"']" ), data))
    labels = np.array([item for sublist in labels for item in sublist])
    source = list(map(lambda tup: eval("['"+tup[2]+"']" ), data))
    source = np.array([item for sublist in source for item in sublist])
    return blocks,labels,source

def text_to_num(blocks,conv_dict):
    """Converts the words to numbers

    blocks: list of lists, every list contains all words of a block
    conv_dict: dictionary, conversion dictionary from text to numbers
    """
    words = list(map(lambda lis: list(map(lambda word: conv_dict[word],lis))\
                     ,blocks))
    return words


def insert_padding(blocks,conv_dict):
    """Returns blocks of same length where text is converted to numbers

    blocks: list of lists, every list contains all words of a block
    conv_dict: dictionary, conversion dictionary from text to numbers

    padding of 0's is added at the end to make them the same length
    """
    #Find length of the longest block and add padding
    max_len = len(max(blocks,key=len))
    print("Sample length: {}".format(max_len))
    blocks = keras.preprocessing.sequence.pad_sequences(blocks,\
                                                value=conv_dict["<PAD>"],\
                                                padding='post',\
                                                maxlen=max_len)
    return blocks,max_len

def create_test_set(blocks, labels, source, mode = 'all', fold = 5, forbidden = {}, test_fraction = None):
    """Returns training,validation and testing data+labels

    blocks: list of lists, every list contains all words of a block
    labels: numpy array of factors as integers
    mode: string, can be 'all', 'cv' or 'split'
    source: numpy array of document names as integers
    OPTIONAL:
    forbidden: dictionary, containing index of documents already used for testing
    test_fraction: float, if mode=='split' fraction of dataset used for testing

    """
    df = pandas.DataFrame(data = list(zip(*[blocks,labels,source])),\
                              columns = ['blocks','labels','source'])
    if mode == 'cv':
        print('Running {} fold cross-validation run {} out of {}.'\
              .format(fold,int(len(forbidden[0])/2)+1,fold))
        test_data = np.array([])
        test_labels = np.array([], dtype = 'int32')
        
        for label in np.unique(df[['labels']]):
            subset = df.loc[(df['labels'] == label)]
            if label == 0:
                n_documents_testing = len(np.unique(subset[['source']]))/fold
                eligable = []
                for docu_index in np.unique(subset[['source']]):
                    if docu_index not in forbidden[0]:
                        eligable.append(docu_index)
                selected_documents = [selected for selected in np.random.choice(eligable,size=2, replace = False)]
                forbidden[0] += selected_documents
            if label == 1:
                n_documents_testing = len(np.unique(subset[['source']]))/fold
                eligable = []
                for docu_index in np.unique(subset[['source']]):
                    if docu_index not in forbidden[1]:
                        eligable.append(docu_index)
                selected_documents = [selected for selected in np.random.choice(eligable,size=2, replace = False)]
                forbidden[1] += selected_documents
            if label == 2:
                n_documents_testing = 2 * len(np.unique(subset[['source']]))/fold
                eligable = []
                for docu_index in np.unique(subset[['source']]):
                    if docu_index not in forbidden[2]:
                        eligable.append(docu_index)
                selected_documents = [selected for selected in np.random.choice(eligable,size=4, replace = False)]
                forbidden[2] += selected_documents
            test_set = df.loc[(df['labels'] == label) & (df['source'].isin(selected_documents))]
            test_data = np.append(test_data,test_set[['blocks']])
            test_labels = np.append(test_labels,test_set[['labels']])
            #remove test set from dataframe
            df = df.drop(test_set.index.values)
        print(forbidden)

        train_data = np.array(list(map(lambda np_lis: np_lis.tolist(),
                                       np.squeeze(np.array(df[['blocks']])))))
        train_labels = np.squeeze(np.array(df[['labels']], dtype = 'int32'))
        test_data = np.array(list(map(lambda np_lis: np_lis.tolist(),test_data)))
        test_labels = np.squeeze(test_labels)
        return train_data, train_labels, \
               test_data, test_labels, forbidden


 
    if mode == 'split':
        print("Splitting data in training ({}) and testing ({}) set"
              .format(1-test_fraction,test_fraction))
        test_mask = np.random.choice([False, True],\
                            len(labels), p=[1-test_fraction, test_fraction])
        
        test_data = blocks[test_mask]
        test_labels = labels[test_mask]

        train_data = blocks[test_mask == False]
        train_labels = labels[test_mask == False]
        
        return train_data, train_labels, \
               test_data, test_labels

    if mode == 'all':
        print("Using all available data for training")
        test_data = np.array([])
        test_labels = np.array([])
        train_data = np.array(list(map(lambda np_lis: np_lis.tolist(),
                                       np.array(df[['blocks']]))))
        train_labels = np.array(df[['labels']], dtype = 'int32')

        return train_data, train_labels, \
               test_data, test_labels

def estimator_spec_for_softmax_classification(logits, labels, mode):
    """Returns EstimatorSpec instance for softmax classification."""
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {'predict_output': tf.estimator.export.PredictOutput\
                                ({"pred_output_classes": predicted_classes,\
                                  'probabilities': tf.nn.softmax(logits)})}
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'class': predicted_classes,
                'prob': tf.nn.softmax(logits)
            }, export_outputs=export_outputs)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        'accuracy':
        tf.metrics.accuracy(labels=labels, predictions=predicted_classes)
          }
    return tf.estimator.EstimatorSpec(
    mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def serving_input_receiver_fn():
    """Prepares the model variables for export"""
    serialized_tf_example = tf.placeholder(dtype=tf.string,\
                                           shape=[None], name='input_tensors')
    receiver_tensors = {"predictor_inputs": serialized_tf_example}
    feature_spec = {WORDS_FEATURE: tf.FixedLenFeature([max_len],tf.int64)}
    features = tf.parse_example(serialized_tf_example, feature_spec)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

def bag_of_words_model(features, labels, mode):
    """A bag-of-words model. Note it disregards the word order in the text."""
    bow_column = tf.feature_column.categorical_column_with_identity(
      WORDS_FEATURE, num_buckets=n_words)
    bow_embedding_column = tf.feature_column.embedding_column(
      bow_column, dimension=EMBEDDING_SIZE)
    bow = tf.feature_column.input_layer(
      features, feature_columns=[bow_embedding_column])
    logits = tf.layers.dense(bow, MAX_LABEL, activation=None)
    return estimator_spec_for_softmax_classification(
            logits=logits, labels=labels, mode=mode)

def labels_to_factors(labels):
    """Returns a numpy array of factors as integers

    labels: numpy array of factors as strings
    """
    levels,ind = np.unique(labels,return_index=True)
    levels = levels[np.argsort(ind)] 
    level_dict = {level:i for (i,level) in enumerate(levels)}
    factors = list(map(lambda label: level_dict[label],labels))
    return np.array(factors)

def run_model(model_dir, mode = 'all', test_franction = None, fold = 5):
    """Runs the training and testing steps of the model

    model_dir: string, directory to store checkpoint variables
    mode: string, can be 'all', 'cv' or 'split'

    Prints accuracy and confusion matrix for the test results
    """

    model_fn = bag_of_words_model
    classifier = tf.estimator.Estimator(model_fn = model_fn,\
                                        model_dir = model_dir)

    if mode == "cv":
        forbidden = {}
        forbidden.setdefault(0,[])
        forbidden.setdefault(1,[])
        forbidden.setdefault(2,[])
        for i in range(fold):
            #initiate cross-validation estimator
            classifier = tf.estimator.Estimator(model_fn = model_fn,\
                                        model_dir = model_dir+'/cv_run_{}'.format(i+1))
            #generate randomized selection of testing/training split
            train_data, train_labels, test_data, test_labels, forbidden =\
                 create_test_set(blocks, labels, source, \
                                 mode,test_fraction = test_franction,\
                                 fold = fold,forbidden=forbidden)
            # Train.
            train_input_fn = tf.estimator.inputs.numpy_input_fn(
              x={WORDS_FEATURE: train_data},
              y=train_labels,
              batch_size=len(train_data),
              num_epochs=None,
              shuffle=True)
            classifier.train(input_fn=train_input_fn, steps=100)
            #Test
            test_input_fn = tf.estimator.inputs.numpy_input_fn(
              x={WORDS_FEATURE: test_data}, y=test_labels, num_epochs=1, shuffle=False)
            predictions = classifier.predict(input_fn=test_input_fn)
            y_predicted = np.array(list(p['class'] for p in predictions))
            y_predicted = y_predicted.reshape(np.array(test_labels).shape)
            print('Testing size: {}, Training size {}'\
                  .format(len(y_predicted),len(train_labels)))
            
            # Score with tensorflow.
            scores = classifier.evaluate(input_fn=test_input_fn)
            accuracy = scores['accuracy']
            print('Accuracy (tensorflow): {0:f}'.format(accuracy))
            cm = tf.contrib.metrics.confusion_matrix(test_labels, y_predicted)
            with tf.Session() as sess:
                confusion_matrix = sess.run(cm)
                df = pandas.DataFrame(confusion_matrix,\
                        columns = ['Adaptation','Mitigation','Non-climate'],\
                        index =  ['Adaptation','Mitigation','Non-climate'])
            save_metadata(model_directory,"a",accuracy,len(y_predicted),len(train_labels),df)
        return classifier,accuracy,df,len(y_predicted),len(train_labels)
    
    elif mode == 'split':
        train_data, train_labels, test_data, test_labels =\
                 create_test_set(blocks, labels, source, \
                                 mode,test_fraction = test_franction)
        # Train.
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
          x={WORDS_FEATURE: train_data},
          y=train_labels,
          batch_size=len(train_data),
          num_epochs=None,
          shuffle=True)
        classifier.train(input_fn=train_input_fn, steps=100)
      # Test.
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
          x={WORDS_FEATURE: test_data}, y=test_labels, num_epochs=1, shuffle=False)
        predictions = classifier.predict(input_fn=test_input_fn)
        y_predicted = np.array(list(p['class'] for p in predictions))
        y_predicted = y_predicted.reshape(np.array(test_labels).shape)
        print('Testing size: {}, Training size {}'\
              .format(len(y_predicted),len(train_labels)))

      # Score with tensorflow.
        scores = classifier.evaluate(input_fn=test_input_fn)
        print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))
        cm = tf.contrib.metrics.confusion_matrix(test_labels, y_predicted)
        with tf.Session() as sess:
            confusion_matrix = sess.run(cm)
            df = pandas.DataFrame(confusion_matrix,\
                    columns = ['Adaptation','Mitigation','Non-climate'],\
                    index =  ['Adaptation','Mitigation','Non-climate'])

        save_metadata(model_directory,accuracy,len(y_predicted),len(train_labels),df, savemode = "w")
        return classifier,scores['accuracy'],df,len(y_predicted),len(train_labels)
    
    elif mode == 'all':
        train_data, train_labels, test_data, test_labels =\
                 create_test_set(blocks, labels, source, \
                                 mode,test_fraction = test_franction)
        # Train.
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
          x={WORDS_FEATURE: train_data},
          y=train_labels,
          batch_size=len(train_data),
          num_epochs=None,
          shuffle=True)
        classifier.train(input_fn=train_input_fn, steps=100)
        print('Model trained with all available data: {}'.format(len(train_labels)))
        save_metadata(model_directory, train_size=len(train_labels), savemode = "w")
        return classifier, None, None, 0,len(train_labels)

def reset_model(model_directory):
    """Removes all files from model_directory"""
    if os.path.exists(model_directory):
        shutil.rmtree(model_directory.rsplit('/',1)[0])
    
def save_metadata(model_directory, savemode, accuracy=None, test_size=None, train_size=None, conf_matrix=None):
    """Saves length of data for use during predictions"""
    filename = 'metadata.txt'
    txt = open(os.path.join(model_directory,filename), mode = savemode)
    txt.write(str(max_len)+'\n')
    txt.write(str(datetime.datetime.now())+'\n')
    txt.write(str(accuracy)+'\n'+'test: {}, train: {}\n'.\
              format(test_size,train_size)+str(conf_matrix))
    txt.close()
    
if __name__ == '__main__':
    #constants and data collecting
    model_directory = os.path.join('..','tensorflow','logdir')
    start_time = time.perf_counter()
    c,conn = connect_to_db("climate.db")
    conv_dict = load_conv_dict("conversion_dictionary.txt")
    blocks,labels,source = collect_data_label(c,'Labeled_data')
    #preparing data for tensorfow
    blocks = text_to_num(blocks,conv_dict)
    blocks,max_len = insert_padding(blocks,conv_dict)
    labels = labels_to_factors(labels)
    source = labels_to_factors(source)
    #parameters for model
    reset_model(model_directory)
    WORDS_FEATURE = 'words'
    n_words = len(conv_dict.keys())
    print("Vocabulary size: {} words".format(n_words))
    EMBEDDING_SIZE = 50
    MAX_LABEL = 3
#     mode = "cv"
    #assess model set mode to cv, split or all.
    classifier,accuracy,conf_matrix,test_size,train_size = \
          run_model(model_directory, mode = 'cv')
#         run_model(model_directory, mode = mode, fold = 5)
    print('Model took {} seconds train'.format(int(time.perf_counter() - start_time)))
    #export model and metadata
    full_model_dir = classifier.export_savedmodel\
                     (export_dir_base=model_directory,\
                      serving_input_receiver_fn=serving_input_receiver_fn)
    print('Model saved in {}'.format(full_model_dir))
    
