#!usr/bin/env/python
"""
Authors: Fatih Uenal & Shashi Badloe
Description: Uses block prediction probabilities to reach a
conclusion on the entire document
Source of the model: https://link.springer.com/article/10.1007%2Fs10113-020-01677-8
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from numberizer import connect_to_db
from pdf_parser import create_folder

def get_document_names(cursor):
    """Returns unique document filenames from the database for unlabeled_data"""
    source = cursor.execute("""SELECT source FROM Unlabeled_data""").fetchall()
    unique_documents = list(set(source))
    return unique_documents

def get_predictions(cursor):
    """Returns dictionary of {filename:predicted_class}"""
    pred_dict = {}
    for document in get_document_names(cursor):
        parameters = (document[0],)
        predictions = cursor.execute("""SELECT pred_class
                                        FROM Unlabeled_data
                                        WHERE source = ?"""\
                                     ,parameters).fetchall()
        pred_dict[document[0]] = list(map(lambda label: label[0],predictions))
    return pred_dict
        
def get_probabilities(cursor):
    """Returns dictionary of {filename:[prob_A,prob_M,prob_N]}"""
    prob_dict = {}
    for document in get_document_names(cursor):
        parameters = (document[0],)
        probabilities = cursor.execute("""SELECT probabilities
                                          FROM Unlabeled_data
                                          WHERE source = ?"""\
                                       ,parameters).fetchall()
        probabilities = list(map(lambda tup: eval("["+tup[0]+"]" ),\
                                 probabilities))
        prob_dict[document[0]] = probabilities
    return prob_dict
            
def plot_histogram(cursor,target_dir):
    labels = ('Fossil-Fuel','Adaptation','Non-climate')
    #histogram with absolute block counts
    pred_dict = get_predictions(cursor)
    pred_count_dict = {}
    for key in sorted(pred_dict.keys()):
        pred_count_dict[key] = [pred_dict[key].count(labels[0]),\
                                pred_dict[key].count(labels[1]),\
                                pred_dict[key].count(labels[2])]
    
    pred_df = pd.DataFrame.from_dict(pred_count_dict,orient = 'index',columns = labels)
    pred_df.plot(kind = 'bar', stacked = True)
    plt.subplots_adjust(bottom=0.25)
    plt.title('Document block predictions')
    plt.xlabel('Document name')
    plt.ylabel('Number of blocks')
    plt.savefig(os.path.join(target_dir,'doc_pred'))
    #histogram with percentages
    prob_dict = get_probabilities(cursor)
    prob_count_dict = {}
    for key in sorted(prob_dict.keys()):
        prob_count_dict[key] = list(map(lambda lis: sum(lis)/len(lis)*100,\
                           zip(*prob_dict[key])))
    prob_df = pd.DataFrame.from_dict(prob_count_dict,orient = 'index',columns = labels)
    prob_df.plot(kind = 'bar', stacked = True)
    plt.subplots_adjust(bottom=0.25)
    plt.title('Document block predictions')
    plt.xlabel('Document name')
    plt.ylabel('Percentage of blocks')
    plt.savefig(os.path.join(target_dir,'doc_prob'))
    
if __name__ == '__main__':
    target_dir = '../Plots'
    create_folder(target_dir)
    c,conn = connect_to_db("climate.db")
    plot_histogram(c,target_dir)
