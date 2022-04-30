#!usr/bin/env/python
"""
Authors: Fatih Uenal & Shashi Badloe
Description: Visualization for fraction of high confidence blocks
amongst documents
Source of the model: https://link.springer.com/article/10.1007%2Fs10113-020-01677-8
"""
import os
import sqlite3
from numberizer import connect_to_db
import matplotlib.pyplot as plt
import numpy as np
from pdf_parser import create_folder
import time

def retrieve_unlabeled(c,table_name):
    """Returns data from Unlabeled_data in the database
    c: database cursor
    table_name: string, name of the table containing unlabeled data
    """
    c.execute('SELECT distinct source FROM {}'.format(table_name))
    docu_names = list(map(lambda tup: tup[0],c.fetchall()))
    for docu_name in docu_names:
        print(docu_name)
        c.execute("SELECT date,dept,original,source,probabilities FROM {} WHERE source = '{}'".format(table_name,docu_name))
        data = c.fetchall()
        yield data

def count_high_prob(data,label,treshold):
    """Returns a list of ints for the count of high probability blocks in docs

    data: tuple, stores (date,original,source,probabilities)
    label: string, one of the three classes the data is predicted on
    treshold: float between 0-1, minumum confidence level
    given a label, count the number of high probability blocks for a label
    for every document
    """
    count_dict = {}
    label_dict = {'Adaptation':0,'Mitigation':1,'Non-climate':2}
    for date,dept,original,source,probabilities in data:
        probabilities = list(map(float,probabilities.replace(' ','').\
                                     split(',')))
        probability = probabilities[label_dict[label]]
        count_dict.setdefault(source,[0,dept,date_to_numbers(date)])
        if probability >= treshold:
            count_dict[source][0] += 1
    count_dict = normalize_count_dict(count_dict,data)
    return count_dict

def normalize_count_dict(count_dict,data):
    """Turns high conf. block counts into percentage of document"""
    for key in count_dict.keys():
        count_dict[key][0] = count_dict[key][0]/sum(list(map(lambda tup:tup.\
                                                       count(key),data)))
    return count_dict

def date_to_numbers(date):
    """Transforms date 'day month year' to format YYYY-MM-DD

    date: string, date i.e. '8 September 2018'
    """
    month_dict = {'January':1,'February':2,'March':3,'April':4,'May':5,\
                  'June':6,'July':7,'August':8,'September':9,'October':10,\
                  'November':11,'December':12}
    day,month,year = date.split()
    month = month_dict[month]
    return '-'.join([str(year),str(month),str(day)])

def write_label_lookup(count_dict,label,i):
    """Writes a tabulated text file with documentnames matched to an index

    x: tuple of strings, strings are document names ordered on relative freq
    y: tuple of floats, corresponding relative abundance of high conf. blocks

    output file is in the target directory and named 'label_lookup.txt'
    """
    lists = sorted(list(count_dict.items()), key = lambda tup: tup[1],\
                   reverse = True)
    x,y = zip(*lists)
    y,dept,date = zip(*y)

    with open(os.path.join(target_dir,outfilename),'a') as label:
        for j,docu_name in enumerate(x):
            label.write('{}\t{}\t{}\t{}\t{}\n'.format(i,docu_name,dept[j],date[j],y[0]))       

def write_header(filename):
    with open(filename,'w') as header:
        header.write('index\tdocument\tdepartment\tdate\thigh_conf\n')
            
def reset_outfile(filename):
    """Removes database with name: db_name"""
    if os.path.isfile(filename):
        os.remove(filename) 
    
if __name__ == '__main__':
    #For what class do you want to plot a high confidence distribution?
    label = 'Adaptation'
    outfilename = '{}_lookup.txt'.format(label)
    #Directory to save plote
    target_dir = '../Plots'
    create_folder(target_dir)
    start_time = time.clock()
    c,conn = connect_to_db("climate.db")
    reset_outfile(os.path.join(target_dir,outfilename))
    write_header(os.path.join(target_dir,outfilename))
    for i,data in enumerate(retrieve_unlabeled(c,'Unlabeled_data'),1):
        count_dict = count_high_prob(data,label,0.8)
        write_label_lookup(count_dict,label,i)
        if i % 1000 == 0:
            print("{} documents completed".format(i))
    
    print("Script finished")
    print('Data took {} seconds to prepare'.format(int(time.clock() - start_time)))

